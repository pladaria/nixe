use super::*;
use crate::residency::tests::{backing_allocation, cpu_read};
pub(super) use crate::test_device as support;
use nixe_gpu::physical::PhysicalMapping;
use nixe_gpu::*;
use std::time::Instant;
fn consume(
    execution: &mut NativeGpuExecution,
    submission: &NativeWork,
) -> Result<(), BackendRuntimeError> {
    let started = Instant::now();
    while !execution.consume(submission)? {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "native compilation must publish readiness"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

fn finish_next(
    execution: &mut NativeGpuExecution,
) -> Result<Option<BackendExecutionCompletion>, VulkanError> {
    execution.residency.flush()?;
    execution.completion(Some(Duration::from_secs(5)))
}

#[test]
fn frame_metrics_follow_resident_producers_and_publish_after_asynchronous_completion() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let families = unsafe {
        device
            .owner
            .instance
            .instance
            .get_physical_device_queue_family_properties(device.owner.physical)
    };
    let timestamps =
        families[device.queues().graphics_compute_family as usize].timestamp_valid_bits != 0;
    let correlated = if cfg!(target_os = "linux") && device.owner.calibrated_timestamps {
        let calibration = ash::ext::calibrated_timestamps::Instance::new(
            &device.owner.instance.entry,
            &device.owner.instance.instance,
        );
        unsafe { calibration.get_physical_device_calibrateable_time_domains(device.owner.physical) }
            .unwrap()
            .contains(&vk::TimeDomainEXT::CLOCK_MONOTONIC)
    } else {
        false
    };
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"frame-time-test!",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let metrics = execution.frame_metrics();
    let (wake, wakes) = std::sync::mpsc::channel();
    execution
        .bind_host_wakeup(Arc::new(move || {
            let _ = wake.send(());
        }))
        .unwrap();
    let (_allocation, mapping) = backing_allocation(&execution.residency, 4096, 4096, 0x51);
    for frame in 1..=4 {
        if frame == 1 {
            // A host/coherency operation may leave a shared slot open before
            // guest command consumption begins. Metrics must not split it.
            execution.residency.device_mut().begin_recording().unwrap();
        }
        let started = Instant::now();
        execution.begin_frame_work();
        let image = execution
            .acquire_presentable_image(
                NativePresentationImage {
                    inputs: mapping.pin(0, mapping.bytes()).unwrap(),
                    planes: Arc::from([GuestImagePlane {
                        depth_slice: 0,
                        mip_level: 0,
                        array_layer: 0,
                        aspect: GuestImageAspect::Color,
                        footprint: crate::residency::tests::plane_geometry(
                            Arc::clone(&mapping),
                            [4, 4],
                            16,
                            4,
                        ),
                        row_pitch: 16,
                        bytes_per_block: 4,
                    }]),
                    extent: [4, 4],
                    format: PresentationImageFormat::Rgba8,
                    allow_canonical_import: true,
                },
                (frame % 3) as u32,
            )
            .unwrap();
        let producer = image.lease().producer().slot();
        assert_eq!(execution.residency.device().scheduled_tick(), producer);
        let deadline = Instant::now() + Duration::from_secs(5);
        let sample = loop {
            execution.progress().unwrap();
            if let Some(sample) = metrics.read_since(frame - 1, 1).samples.first().copied() {
                break sample;
            }
            execution.arm_completion_wakeup().unwrap();
            wakes
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap();
        };
        assert_eq!(sample.sequence, frame);
        assert_eq!(sample.producer, producer);
        assert_eq!(sample.swap_interval, (frame % 3) as u32);
        assert!(!sample.cpu.is_zero());
        assert_eq!(sample.gpu.is_some(), timestamps);
        assert_eq!(sample.total.is_some(), timestamps && correlated);
        assert_eq!(sample.clock_error.is_some(), timestamps && correlated);
        if let (Some(total), Some(error)) = (sample.total, sample.clock_error) {
            assert!(total <= started.elapsed() + error);
            assert!(sample.gpu.unwrap() <= total + error);
        }
        assert!(
            sample
                .total
                .is_none_or(|total| total <= Duration::from_secs(5))
        );
        assert_eq!(
            execution.residency.device().scheduled_tick(),
            producer,
            "reading metrics must not submit native work"
        );
    }
    execution.teardown().unwrap();
    drop(execution);
    assert_eq!(metrics.read_since(0, 10).samples.len(), 4);
    monitor.check().unwrap();
}

fn captured_vertex_fragment(
    configuration: &nixe_gpu::shader_compilation::Configuration,
) -> Arc<GuestShaderCode> {
    captured_graphics(configuration, None)
}

fn captured_graphics(
    configuration: &nixe_gpu::shader_compilation::Configuration,
    fragment: Option<([u32; 20], Vec<u64>)>,
) -> Arc<GuestShaderCode> {
    use nixe_gpu::shader_compilation::{ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Compiler, Stage,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        maxwell::header::ProgramHeader,
        types::ProgramOffset,
    };
    let attribute = |store: bool, offset: u64| {
        (if store {
            0xeff0_0000_0000_0000
        } else {
            0xefd8_0000_0000_0000
        }) | (3 << 47)
            | (255 << 39)
            | (offset << 20)
            | (7 << 16)
            | (255 << 8)
    };
    let exit = 0xe300_0000_0007_000f;
    let (fragment_header, fragment) = fragment.unwrap_or_else(|| {
        let mut header = [0; 20];
        header[6] = 0xaa;
        header[18] = 15;
        (
            header,
            (0..4u64)
                .map(|component| 0xe000_0000_0007_ff00 | ((32 + component) << 30) | component)
                .chain([exit])
                .collect(),
        )
    });
    let mut compiler = Compiler::new().unwrap();
    let mut stages = Vec::new();
    for (stage, instructions) in [
        (
            Stage::VertexB,
            vec![
                attribute(false, 0x80),
                attribute(true, 0x70),
                attribute(false, 0x90),
                attribute(true, 0x80),
                exit,
            ],
        ),
        (Stage::Fragment, fragment),
    ] {
        let header = if stage == Stage::Fragment {
            fragment_header
        } else {
            [0; 20]
        };
        let bytes: [u8; 80] = header
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut pc = 0x50;
        let mut environment = SnapshotEnvironment::new(
            EnvironmentMetadata {
                stage,
                sph: ProgramHeader::from_bytes(&bytes),
                ..Default::default()
            },
            Observations {
                instructions: instructions
                    .into_iter()
                    .map(|word| {
                        if pc % 32 == 0 {
                            pc += 8;
                        }
                        let address = ProgramOffset(pc);
                        pc += 8;
                        (address, word)
                    })
                    .collect(),
                viewport_transform_state: Some(1),
                ..Default::default()
            },
        );
        let id = compiler
            .translate(
                &mut environment,
                &configuration.host,
                &configuration.settings,
            )
            .unwrap();
        stages.push(StageSource {
            environment: Arc::new(environment),
            info: Arc::new(compiler.program(id).unwrap().info.clone()),
            binding_group: Some(0),
        });
    }
    Arc::new(ShaderInputs::new(stages.into(), configuration))
}

#[test]
fn unknown_cpu_readback_leaves_recording_available_and_preserves_its_version() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"async-page-test!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, backing) = backing_allocation(&execution.residency, 4096, 4096, 0x25);
    let id = execution
        .residency
        .buffer_for(backing.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap()
        .buffer;
    execution.residency.write_buffer(id, 9, &[0x43; 4]).unwrap();
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
    execution
        .bind_host_wakeup(Arc::new(move || {
            let _ = wake.send(());
        }))
        .unwrap();
    let (request, mut pending) = execution
        .residency
        .memory()
        .prepare_device_access(
            backing.pin(0, backing.bytes()).unwrap(),
            nixe_memory::AccessOperation::Read(Vec::new()),
        )
        .unwrap();
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
    execution.queue_device_access(request).unwrap();
    assert!(pending.poll().is_none());
    assert!(
        execution.residency.readback_point().unwrap()
            <= execution.residency.device().scheduled_tick(),
        "a CPU demand must submit its download before unrelated recording"
    );
    let batch = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::WaitForIdle,
    };

    consume(&mut execution, &batch)
        .and_then(|()| execution.submit(batch.frontend))
        .unwrap();
    assert!(
        execution.residency.buffer_visibility_work().is_none(),
        "physical submission must not duplicate an exact CPU demand as a publication download"
    );
    let submitted = execution.residency.device().scheduled_tick();
    execution.progress().unwrap();
    assert!(pending.poll().is_none());
    assert_eq!(
        execution.residency.device().scheduled_tick(),
        submitted,
        "retrying an already submitted demand must not split independent recording"
    );
    execution.arm_completion_wakeup().unwrap();
    release.send(()).unwrap();
    wakes.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(
        watchdog.join().unwrap(),
        "unknown GPU data blocked independent recording"
    );
    let deadline = Instant::now() + Duration::from_secs(3);
    let result = loop {
        execution.progress().unwrap();
        if let Some(result) = pending.poll() {
            let nixe_memory::AccessResult::Read(bytes) = result.unwrap() else {
                panic!("expected read")
            };
            break bytes;
        }
        wakes
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
    };
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
    assert_eq!(cpu_read(&mut execution.residency, &backing), bytes);
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
            &mut device.dispatch,
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
    )
    .unwrap();
    observer.arm(later.tick()).unwrap();
    observer.arm(first.tick()).unwrap();
    let earlier_notified = notifications.recv_timeout(Duration::from_secs(1)).is_ok();
    observer.arm(first.tick()).unwrap();
    let completed_notified = notifications.recv_timeout(Duration::from_secs(1)).is_ok();
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
        completed_notified,
        "an already reached action must run immediately"
    );
    assert!(
        later_notified,
        "the later request must remain armed after the earlier wake"
    );
}

#[test]
fn completed_native_work_still_wakes_its_pending_logical_completion() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"completed-wakeup",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, whole) = backing_allocation(&execution.residency, 4096, 4096, 0x19);
    let submission = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::ReportMemory {
            fence: true,
            target: whole.pin(0, whole.bytes()).unwrap(),
            offset: 64,
            bytes: Arc::from([0x37; 16]),
        },
    };
    let (notify, notifications) = std::sync::mpsc::channel();
    execution
        .bind_host_wakeup(Arc::new(move || {
            let _ = notify.send(());
        }))
        .unwrap();

    let token = consume(&mut execution, &submission)
        .and_then(|()| execution.submit(submission.frontend))
        .unwrap();
    let work = execution.completions.front().unwrap().work.unwrap();
    execution.residency.flush().unwrap();
    assert!(
        execution
            .residency
            .device_mut()
            .wait_for_work(work, Duration::from_secs(5))
            .unwrap()
    );
    // Native progress can advance after the owner's poll while the logical
    // completion is still queued. Registration must deliver that action.
    execution.arm_completion_wakeup().unwrap();
    let notified = notifications.recv_timeout(Duration::from_secs(1)).is_ok();
    let completed = execution.poll_completion().unwrap().unwrap();
    assert_eq!(completed.submission(), token);
    drop(execution);
    monitor.check().unwrap();
    assert!(
        notified,
        "completed native work must not strand its logical owner"
    );
}

#[test]
fn native_ordered_writes_publish_known_bytes_without_waiting_for_gpu_only_work() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"ordered-effects!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, whole) = backing_allocation(&execution.residency, 4096, 4096, 0x19);
    let target = whole.slice(64, 16).unwrap();
    let mut kind =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let semaphore = unsafe {
        owner.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    let submission = |frontend, value| {
        NativeWork {
            frontend: FrontendSubmissionId::new(frontend),
            command: NativeCommand::Sequence(
                vec![
                    NativeCommand::ReportMemory {
                        fence: true,
                        target: whole.pin(0, whole.bytes()).unwrap(),
                        offset: 128,
                        bytes: Arc::from([0xa3; 16]),
                    },
                    NativeCommand::WaitForIdle,
                    NativeCommand::ReportMemory {
                        fence: true,
                        target: whole.pin(0, whole.bytes()).unwrap(),
                        offset: 64,
                        bytes: Arc::from([value ^ 0xff; 16]),
                    },
                    // A later release supersedes only the same exact bytes.
                    NativeCommand::ReportMemory {
                        fence: true,
                        target: whole.pin(0, whole.bytes()).unwrap(),
                        offset: 64,
                        bytes: Arc::from([value; 16]),
                    },
                ]
                .into_boxed_slice(),
            ),
        }
    };
    let (woken, wakeups) = std::sync::mpsc::channel();
    execution
        .bind_host_wakeup(Arc::new(move || {
            let _ = woken.send(());
        }))
        .unwrap();
    let first = submission(1, 0x37);

    consume(&mut execution, &first).unwrap();
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
    execution.submit(first.frontend).unwrap();
    execution.arm_completion_wakeup().unwrap();
    assert!(execution.poll_completion().unwrap().is_none());

    // The GPU cannot complete the first batch yet. An unrelated subsequent
    // recording still succeeds; no completion wait is needed for these writes.
    let second = submission(2, 0x71);

    consume(&mut execution, &second)
        .and_then(|()| execution.submit(second.frontend))
        .unwrap();
    assert!(execution.poll_completion().unwrap().is_none());
    let (request, mut pending) = execution
        .residency
        .memory()
        .prepare_device_access(
            whole.pin(0, whole.bytes()).unwrap(),
            nixe_memory::AccessOperation::Read(Vec::new()),
        )
        .unwrap();
    execution.queue_device_access(request).unwrap();
    assert!(
        pending.poll().is_some(),
        "known CPU payloads do not depend on unrelated GPU-only work"
    );
    let independent = NativeWork {
        frontend: FrontendSubmissionId::new(90),
        command: NativeCommand::WaitForIdle,
    };

    consume(&mut execution, &independent)
        .and_then(|()| execution.submit(independent.frontend))
        .unwrap();
    let bytes = cpu_read(&mut execution.residency, &whole);
    assert_eq!(&bytes[..64], &[0x19; 64]);
    assert_eq!(&bytes[64..80], &[0x71; 16]);
    assert!(bytes[80..128].iter().all(|&b| b == 0x19));
    assert_eq!(&bytes[128..144], &[0xa3; 16]);
    assert!(bytes[144..].iter().all(|&b| b == 0x19));
    assert_eq!(
        execution.residency.statistics().native_readback_bytes,
        0,
        "known payload must not perform a redundant download"
    );
    unsafe {
        owner.device.signal_semaphore(
            &vk::SemaphoreSignalInfo::default()
                .semaphore(semaphore)
                .value(1),
        )
    }
    .unwrap();
    wakeups
        .recv_timeout(Duration::from_secs(1))
        .expect("an idle owner must wake without another submission");
    while finish_next(&mut execution).unwrap().is_some() {}
    let id = execution
        .residency
        .buffer_for_deferred_import(target, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    assert_eq!(
        execution
            .residency
            .read_buffer(id.buffer, id.offset, 16)
            .unwrap(),
        [0x71; 16],
        "the real GPU bytes must agree with canonical publication"
    );
    // A different native producer invalidates the known payload. Its bytes must
    // be read from the GPU rather than replaying the preceding host value.
    execution
        .residency
        .write_buffer(id.buffer, id.offset, &[0xa7; 16])
        .unwrap();

    assert_eq!(
        &cpu_read(&mut execution.residency, &whole)[64..80],
        &[0xa7; 16]
    );
    // Physical delivery still follows the real queue prefix. A CPU-authored
    // release must not invent an empty native submission.
    let prefix = NativeWork {
        frontend: FrontendSubmissionId::new(3),
        command: NativeCommand::ReportMemory {
            fence: true,
            target: whole.pin(0, whole.bytes()).unwrap(),
            offset: 64,
            bytes: Arc::from([0xb3; 16]),
        },
    };

    let uploaded_before_release = execution.residency.statistics().cpu_uploaded_bytes;
    consume(&mut execution, &prefix)
        .and_then(|()| execution.submit(prefix.frontend))
        .unwrap();
    let released_tick = execution.completions.back().unwrap().work.unwrap().tick();
    assert_eq!(
        execution.residency.statistics().cpu_uploaded_bytes,
        uploaded_before_release,
        "a CPU-only completion effect must not upload its known payload"
    );
    let following = NativeWork {
        frontend: FrontendSubmissionId::new(4),
        command: NativeCommand::ReportMemory {
            fence: true,
            target: whole.pin(0, whole.bytes()).unwrap(),
            offset: 128,
            bytes: Arc::from([0xc3; 16]),
        },
    };

    consume(&mut execution, &following)
        .and_then(|()| execution.submit(following.frontend))
        .unwrap();
    assert!(execution.completions.back().unwrap().work.unwrap().tick() >= released_tick);
    let source = execution
        .residency
        .buffer_for_deferred_import(
            whole.slice(64, 16).unwrap(),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    let (_copied, copied) = backing_allocation(&execution.residency, 16, 16, 0);
    let regions = [nixe_gpu::native::BufferCopyRegion {
        source: 64,
        destination: 0,
        bytes: 16,
    }];
    execution
        .residency
        .copy_physical_buffers(&whole, &copied, &regions)
        .unwrap();
    let uploaded = execution.residency.statistics().cpu_uploaded_bytes;
    execution
        .residency
        .copy_physical_buffers(&whole, &copied, &regions)
        .unwrap();
    assert_eq!(
        execution.residency.statistics().cpu_uploaded_bytes,
        uploaded,
        "repeated native consumers reuse the materialized payload and page baseline"
    );
    execution
        .residency
        .write_buffer(source.buffer, source.offset + 4, &[0xd5; 4])
        .unwrap();
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
    let shader = captured_vertex_fragment(&device.profile().shader_compiler);
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration {
            ..CacheConfiguration::default()
        },
        4 * 1024 * 1024,
    )
    .unwrap();
    let (pixels, backing) = backing_allocation(&execution.residency, 256, 256, 0);
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
        view_identity: None,
        sample_grid: [1, 1],
        description,
        swizzle: Swizzle::IDENTITY,
        canonical_texels: true,
        planes: Arc::from([GuestImagePlane {
            depth_slice: 0,
            mip_level: 0,
            array_layer: 0,
            aspect: GuestImageAspect::Color,
            footprint: crate::residency::tests::plane_geometry(backing.clone(), [8, 8], 32, 4),
            row_pitch: 32,
            bytes_per_block: 4,
        }]),
    };
    let (vertices, vertex_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    let vertex_backing = vertex_backing.slice(0, 72).unwrap();
    let words: [f32; 18] = [
        -1., -1., 0., 0., 1., 0., 3., -1., 0., 0., 1., 0., -1., 3., 0., 0., 1., 0.,
    ];
    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &vertices,
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
            shader,
            bindings: Box::new([]),
            shared_memory_bytes: 0,
        }),
        vertices: Arc::from([GuestVertexBuffer {
            binding: 0,
            backing: vertex_backing,
        }]),
        index: None,
        colors: Arc::from([image]),
        depth_stencil: None,
        parameters: Arc::new(GuestDrawParameters {
            shader_state: Default::default(),
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
    // Dynamic-state support affects selection before shader/pipeline lookup.
    for dynamic in [false, true] {
        use std::hash::{Hash, Hasher};
        let key = GraphicsKey::new(&state, dynamic).unwrap();
        let mut changed = (*state).clone();
        let p = Arc::make_mut(&mut changed.parameters);
        p.scissor.x = 1;
        p.blend_constants = [0.25; 4];
        p.render_area.width = 4;
        let same = GraphicsKey::new(&changed, dynamic).unwrap();
        assert!(key == same);
        let hash = |key: &GraphicsKey| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&key), hash(&same));
        Arc::make_mut(&mut changed.parameters).cull = CullMode::Back;
        assert_eq!(key == GraphicsKey::new(&changed, dynamic).unwrap(), dynamic);
        Arc::make_mut(&mut changed.parameters).vertices[0].stride += 4;
        assert!(key != GraphicsKey::new(&changed, dynamic).unwrap());
    }
    // Scaled integer legalization can share a native fetch format while its
    // shader interpretation differs. The static key must retain that distinction.
    let mut formats = std::collections::HashSet::new();
    for format in [
        VertexFormat::Float32x3,
        VertexFormat::Uint32x3,
        VertexFormat::Uscaled {
            width: VertexComponentWidth::Bits32,
            components: VertexComponentCount::Three,
        },
        VertexFormat::Sscaled {
            width: VertexComponentWidth::Bits32,
            components: VertexComponentCount::Three,
        },
    ] {
        let mut variant = (*state).clone();
        Arc::make_mut(&mut variant.parameters).vertices[0].attributes[0].format = format;
        assert!(formats.insert(GraphicsKey::new(&variant, true).unwrap()));
    }
    let draw = NativeCommand::Draw {
        state: Arc::clone(&state),
        arguments: DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    };
    let submission = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::Sequence(vec![draw; 16].into_boxed_slice()),
    };

    let (_unrelated_allocation, unrelated) =
        backing_allocation(&execution.residency, 128 * 1024, 4096, 0);
    let old = execution
        .residency
        .buffer_for_deferred_import(
            unrelated.slice(0, 4096).unwrap(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
        .unwrap();
    for frontend in 1..=2 {
        if frontend == 2 {
            let merged = execution
                .residency
                .buffer_for_deferred_import(
                    Arc::clone(&unrelated),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )
                .unwrap();
            assert_ne!(merged.buffer, old.buffer);
            assert!(!execution.residency.buffer_binding_current(old.buffer));
        }
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(frontend),
            command: submission.command.clone(),
        };

        consume(&mut execution, &submission)
            .and_then(|()| execution.signal_fence(submission.frontend))
            .unwrap();
        assert_eq!(
            finish_next(&mut execution).unwrap().unwrap().frontend(),
            submission.frontend
        );
        let stats = execution.recording_statistics();
        assert_eq!(stats.pipeline_binds, frontend);
        assert_eq!(stats.vertex_binds, frontend);
        assert_eq!(stats.graphics_updates, frontend);
        assert_eq!(stats.rendering_scopes, frontend);
    }
    assert_eq!(execution.statistics.prepared_draws, 1);
    // Identical recompiled code can have a new allocation after cold-cache
    // eviction. Content identity, not its allocation, selects graphics state.
    let mut recaptured = (*state).clone();
    let program = Arc::make_mut(&mut recaptured.program);
    program.shader = Arc::new((*program.shader).clone());
    let prepared_graphics = execution.statistics.prepared_graphics;
    assert!(execution.graphics(&Arc::new(recaptured)).unwrap());
    assert_eq!(execution.statistics.prepared_graphics, prepared_graphics);
    assert_eq!(execution.statistics.prepared_draws, 1);
    assert_eq!(execution.statistics.draws, 32);
    assert_eq!(execution.statistics.warm_draws, 31);
    let retained_push = Arc::clone(
        &execution.graphics_pipelines[execution.current_graphics.unwrap().0]
            .parameters
            .as_ref()
            .unwrap()
            .2,
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|p| p == [0, 255, 0, 255]));
    assert_eq!(
        execution.residency.statistics().cpu_streamed_bytes,
        0,
        "vertices use retained buffers"
    );
    // A logical fence need not submit GPU-only reads. A following CPU write
    // is uploaded after that snapshot, with a native WAR edge.
    let prefix = NativeWork {
        frontend: FrontendSubmissionId::new(3),
        command: submission.command.clone(),
    };

    let first = consume(&mut execution, &prefix)
        .and_then(|()| execution.signal_fence(prefix.frontend))
        .unwrap();
    assert_eq!(
        first.instance(),
        execution
            .residency
            .device()
            .submission
            .ticket(0)
            .submission()
            .instance()
    );
    let mut red = words;
    for color in red.chunks_exact_mut(6) {
        color[3] = 1.;
        color[4] = 0.;
    }
    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &vertices,
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
    let final_segment = NativeWork {
        frontend: prefix.frontend,
        command: NativeCommand::Draw {
            state: Arc::new(changed),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        },
    };

    let selections = execution.statistics.reused_vertex_selections;
    let second = consume(&mut execution, &final_segment)
        .and_then(|()| execution.signal_fence(final_segment.frontend))
        .unwrap();
    assert!(!Arc::ptr_eq(
        &execution.graphics_pipelines[execution.current_graphics.unwrap().0]
            .parameters
            .as_ref()
            .unwrap()
            .2,
        &retained_push
    ));
    assert!(execution.statistics.reused_vertex_selections > selections);
    assert_ne!(first, second);
    let a = finish_next(&mut execution).unwrap().unwrap();
    let b = finish_next(&mut execution).unwrap().unwrap();
    assert_eq!(a.submission(), first);
    assert_eq!(b.submission(), second);
    assert_eq!(a.submission().instance(), b.submission().instance());
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
    let pending = NativeWork {
        frontend: FrontendSubmissionId::new(4),
        command: submission.command.clone(),
    };

    let submitted = execution.residency.device().queue_usage().submitted_tick;
    let token = consume(&mut execution, &pending)
        .and_then(|()| execution.signal_fence(pending.frontend))
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted,
        "acceptance alone need not submit the native command buffer"
    );
    // Cache eviction cannot release references still needed by the recording
    // worker/GPU. The next use reconstructs the selection with the same source.
    let retained_sources = Arc::strong_count(&state);
    let prepared = execution.drawings.remove(&state.identity).unwrap();
    assert_eq!(prepared.source.as_ptr(), Arc::as_ptr(&state));
    drop(prepared);
    assert_eq!(Arc::strong_count(&state), retained_sources);

    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &vertices,
        0,
        &words
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let repeated = consume(&mut execution, &pending)
        .and_then(|()| execution.signal_fence(pending.frontend))
        .unwrap();
    assert_eq!(
        execution.residency.device().queue_usage().submitted_tick,
        submitted
    );
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    assert!(execution.residency.device().queue_usage().submitted_tick > submitted);
    assert_eq!(
        finish_next(&mut execution).unwrap().unwrap().submission(),
        token
    );
    assert_eq!(
        finish_next(&mut execution).unwrap().unwrap().submission(),
        repeated
    );
    // Different bindings and scissors across frontend deliveries share native
    // rendering, but preserve draw order and their own prepared semantics.
    // Force eviction while every frontend source is still alive and earlier
    // draws remain in the same unsubmitted rendering scope.
    execution.drawings = nixe_gpu::WorkingSet::new(2, usize::MAX);
    let (other_vertices, other_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &other_vertices,
        0,
        &red.into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let other_backing = other_backing.slice(0, 72).unwrap();
    let mut other = (*state).clone();
    other.identity = PreparedStateId::new(10, 1);
    Arc::make_mut(&mut other.vertices)[0].backing = other_backing;
    let other = Arc::new(other);
    let mut clipped = (*other).clone();
    clipped.identity = PreparedStateId::new(11, 1);
    let parameters = Arc::make_mut(&mut clipped.parameters);
    parameters.identity = PreparedStateId::new(12, 1);
    parameters.scissor.width = 4;
    let clipped = Arc::new(clipped);
    let selections = execution.statistics.reused_vertex_selections;
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    let mut rebound = (*state).clone();
    rebound.identity = PreparedStateId::new(13, 1);
    let rebound = Arc::new(rebound);
    let mut retained_push = None;
    for (index, current) in [&rebound, &other, &state, &clipped].into_iter().enumerate() {
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(10 + index as u64),
            command: NativeCommand::Draw {
                state: Arc::clone(current),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            },
        };

        sent.push(
            consume(&mut execution, &submission)
                .and_then(|()| execution.signal_fence(submission.frontend))
                .unwrap(),
        );
        assert!(execution.drawings.usage().0 <= 2);
        let push = &execution.graphics_pipelines[execution.current_graphics.unwrap().0]
            .parameters
            .as_ref()
            .unwrap()
            .2;
        if index == 0 {
            retained_push = Some(Arc::clone(push));
        } else {
            assert_eq!(
                Arc::ptr_eq(push, retained_push.as_ref().unwrap()),
                index != 3,
                "buffer bindings retain native parameters; scissor changes replace them"
            );
        }
    }
    assert_eq!(
        execution.recording_statistics().rendering_scopes - scopes,
        1
    );
    assert!(execution.statistics.reused_vertex_selections > selections);
    assert!(execution.statistics.reused_graphics > 0);
    assert!(execution.statistics.reused_programs > 0);
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
            finish_next(&mut execution).unwrap().unwrap().submission(),
            token
        );
    }
    // CPU updates between accepted draws must preserve distinct read snapshots,
    // including shader descriptors, without ending rendering or submitting early.
    let (uniform, uniform_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    let uniform_backing = uniform_backing.slice(0, 16).unwrap();
    let encode = |rgba: [f32; 4]| {
        rgba.into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut uniform_draw = (*state).clone();
    uniform_draw.identity = PreparedStateId::new(20, 1);
    uniform_draw.program = Arc::new(GuestProgram {
        identity: PreparedStateId::new(21, 1),
        shader: uniform_color_shader(&execution.device_profile().shader_compiler),
        bindings: Box::new([(
            uniform_color_resource(),
            GuestBinding::Buffer(uniform_backing.clone()),
        )]),
        shared_memory_bytes: 0,
    });
    let uniform_draw = Arc::new(uniform_draw);
    // Parameter identity is independent of shader identity. Preparing an
    // alternative shader with the same parameters must retain both variants.
    let variant = NativeWork {
        frontend: FrontendSubmissionId::new(19),
        command: NativeCommand::Draw {
            state: Arc::clone(&uniform_draw),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        },
    };

    consume(&mut execution, &variant).unwrap();
    execution.residency.flush().unwrap();
    assert!(execution.graphics(&state).unwrap());
    let pipeline_count = execution.pipelines.usage().0;
    let graphics_count = execution.statistics.prepared_graphics;
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
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(20 + index as u64),
            command: NativeCommand::Draw {
                state: Arc::new(draw),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            },
        };

        inputs.push((submission, encode(rgba)));
    }
    // Dynamic scissor changes and runtime uniform data retain the native
    // executable; only the recording parameters and buffer visibility change.
    let selections = execution.statistics.reused_vertex_selections;
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut sent = Vec::new();
    for (submission, bytes) in &inputs {
        crate::residency::tests::cpu_write(&mut execution.residency, &uniform, 0, bytes).unwrap();
        sent.push(
            consume(&mut execution, submission)
                .and_then(|()| execution.signal_fence(submission.frontend))
                .unwrap(),
        );
    }
    assert!(execution.statistics.reused_vertex_selections >= selections + inputs.len() as u64);
    assert_eq!(execution.pipelines.usage().0, pipeline_count);
    assert_eq!(execution.statistics.prepared_graphics, graphics_count);
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
            finish_next(&mut execution).unwrap().unwrap().submission(),
            token
        );
    }
    // Reusing immutable preparations must not pin temporary descriptor payloads
    // into the persistent pool. Completed slots recycle them repeatedly.
    let updates = execution.descriptors.statistics().updates;
    let repeated = &inputs[2].0;
    for _ in 0..12 {
        consume(&mut execution, repeated)
            .and_then(|()| execution.signal_fence(repeated.frontend))
            .unwrap();
        finish_next(&mut execution).unwrap().unwrap();
    }
    assert_eq!(execution.descriptors.statistics().updates, updates);
    assert_eq!(execution.pipelines.usage().0, pipeline_count);
    assert!(execution.recording_statistics().descriptor_payload_captures > 0);
    // Streaming has not acknowledged the persistent buffer's dirty observer.
    let native_uniform = execution
        .residency
        .buffer_for(uniform_backing, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer;
    assert_eq!(
        execution
            .residency
            .read_buffer(native_uniform, 0, 16)
            .unwrap(),
        encode([0., 0., 1., 1.])
    );
    // A following GPU writer replaces CPU authority and must bypass CPU streams.
    execution
        .residency
        .write_buffer(native_uniform, 0, &encode([1., 1., 0., 1.]))
        .unwrap();

    let submission = NativeWork {
        frontend: FrontendSubmissionId::new(30),
        command: NativeCommand::Draw {
            state: Arc::clone(&uniform_draw),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        },
    };

    consume(&mut execution, &submission)
        .and_then(|()| execution.signal_fence(submission.frontend))
        .unwrap();
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(
        bytes
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 255, 0, 255])
    );
    finish_next(&mut execution).unwrap().unwrap();
    // Reusing a draw for many effects must not age out its graphics pipeline.
    // A new immutable draw selecting the same graphics state stays warm too.
    let prepared_graphics = execution.statistics.prepared_graphics;
    for _ in 0..600 {
        consume(&mut execution, &submission).unwrap();
    }
    let key =
        GraphicsKey::new(&uniform_draw, execution.device_profile().dynamic_state.base).unwrap();
    assert!(matches!(
        execution.graphics.get(&key),
        Some(Graphics::Ready(_))
    ));
    let mut next_draw = (*uniform_draw).clone();
    next_draw.identity = PreparedStateId::new(5000, 1);
    let next_work = NativeWork {
        frontend: FrontendSubmissionId::new(5000),
        command: NativeCommand::Draw {
            state: Arc::new(next_draw),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            },
        },
    };
    consume(&mut execution, &next_work).unwrap();
    assert_eq!(execution.statistics.prepared_graphics, prepared_graphics);
    assert!(
        cpu_read(&mut execution.residency, &backing)
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 255, 0, 255])
    );
    // Retained guest preparations do not pin native descriptor sets; changing
    // payloads use the growing bank without splitting or repeating any draw.
    let mut retained = Vec::new();
    let before = execution.statistics.draws;
    for index in 0..32 {
        let (allocation, backing) = backing_allocation(&execution.residency, 256, 256, 0);
        crate::residency::tests::cpu_write(
            &mut execution.residency,
            &allocation,
            0,
            &encode([0., 1., 0., 1.]),
        )
        .unwrap();
        let mut source = (*uniform_draw).clone();
        source.identity = PreparedStateId::new(1000 + index, 1);
        let mut program = (*source.program).clone();
        program.identity = PreparedStateId::new(2000 + index, 1);
        program.bindings = Box::new([(
            uniform_color_resource(),
            GuestBinding::Buffer(backing.slice(0, 16).unwrap()),
        )]);
        source.program = Arc::new(program);
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(1000 + index),
            command: NativeCommand::Draw {
                state: Arc::new(source),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            },
        };

        consume(&mut execution, &submission)
            .and_then(|()| execution.signal_fence(submission.frontend))
            .unwrap();
        retained.push((allocation, submission));
    }
    assert_eq!(execution.statistics.draws - before, 32);
    assert_eq!(execution.descriptors.statistics().allocated, 0);
    let bytes = cpu_read(&mut execution.residency, &backing);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    drop(retained);
    constant_blend_factors_follow_dynamic_constants(&mut execution, &state, &backing);
    overlapping_graphics_buffers_preserve_shader_bindings(&mut execution, &uniform_draw, &backing);
    render_area_preserves_undefined_padding(&mut execution, &state);
    window_depth_reconstructs_reads_and_clamps_exports(&mut execution, &state);
    contiguous_instances_preserve_vertex_inputs(&mut execution, &state, &backing);
    pipeline_switches_retain_parameters_and_known_transitions(
        &mut execution,
        &state,
        &uniform_draw,
    );
    attachment_clears_fold_into_loads_and_preserve_active_rendering(
        &mut execution,
        &state,
        &backing,
    );
    idle_queue_executes_prefixes_before_the_command_list_ends(&mut execution, &state, &backing);
    drop(pixels);
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

fn idle_queue_executes_prefixes_before_the_command_list_ends(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
    backing: &PhysicalMapping,
) {
    execution.bind_host_wakeup(Arc::new(|| {})).unwrap();
    let draw = NativeCommand::Draw {
        state: Arc::clone(state),
        arguments: DrawArguments::NonIndexed {
            first_vertex: 0,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        },
    };
    consume(
        execution,
        &NativeWork {
            frontend: FrontendSubmissionId::new(10000),
            command: draw.clone(),
        },
    )
    .unwrap();
    execution.end_command_list().unwrap();
    let expected = cpu_read(&mut execution.residency, backing);
    let before = execution.residency.device().scheduled_tick();
    let completions = execution.completions.len();
    let draws = execution.statistics.draws;
    // Include a clear at a batch boundary and a draw beyond the last boundary.
    // Prefix submission must neither invent a fence nor omit the trailing work.
    let mut commands = vec![draw.clone(); 63];
    commands.push(NativeCommand::Clear {
        image: state.colors[0].clone(),
        rectangle: state.parameters.render_area,
        value: ClearValue::Color([1., 0., 0., 1.]),
    });
    commands.extend(std::iter::repeat_n(draw, 193));
    consume(
        execution,
        &NativeWork {
            frontend: FrontendSubmissionId::new(10001),
            command: NativeCommand::Sequence(commands.into_boxed_slice()),
        },
    )
    .unwrap();
    assert!(execution.residency.device().scheduled_tick() > before);
    assert_eq!(execution.statistics.draws - draws, 256);
    assert_eq!(execution.completions.len(), completions);
    execution.end_command_list().unwrap();
    assert_eq!(cpu_read(&mut execution.residency, backing), expected);
}

fn attachment_clears_fold_into_loads_and_preserve_active_rendering(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
    backing: &PhysicalMapping,
) {
    let image = state.colors[0].clone();
    let area = state.parameters.render_area;
    execution.residency.device.end_rendering();
    let clear = |color| NativeWork {
        frontend: FrontendSubmissionId::new(9000),
        command: NativeCommand::Clear {
            image: image.clone(),
            rectangle: area,
            value: ClearValue::Color(color),
        },
    };
    let scopes = execution.recording_statistics().rendering_scopes;
    consume(execution, &clear([1., 0., 0., 1.])).unwrap();
    consume(execution, &clear([0., 1., 0., 1.])).unwrap();
    assert_eq!(execution.recording_statistics().rendering_scopes, scopes);
    let mut draw = NativeWork {
        frontend: FrontendSubmissionId::new(9001),
        command: NativeCommand::Draw {
            state: Arc::clone(state),
            arguments: DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 0,
                first_instance: 0,
                instance_count: 1,
            },
        },
    };
    consume(execution, &draw).unwrap();
    assert_eq!(
        execution.recording_statistics().rendering_scopes,
        scopes + 1
    );
    assert!(
        cpu_read(&mut execution.residency, backing)
            .chunks_exact(4)
            .all(|p| p == [0, 255, 0, 255])
    );
    if let NativeCommand::Draw {
        arguments: DrawArguments::NonIndexed { vertex_count, .. },
        ..
    } = &mut draw.command
    {
        *vertex_count = 3;
    }
    consume(execution, &draw).unwrap();
    let scopes = execution.recording_statistics().rendering_scopes;
    let mut partial = clear([1., 0., 0., 1.]);
    if let NativeCommand::Clear { rectangle, .. } = &mut partial.command {
        rectangle.width /= 2;
    }
    consume(execution, &partial).unwrap();
    assert_eq!(execution.recording_statistics().rendering_scopes, scopes);
    let pixels = cpu_read(&mut execution.residency, backing);
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel,
            if index % (area.width as usize) < area.width as usize / 2 {
                &[255, 0, 0, 255]
            } else {
                &[0, 255, 0, 255]
            }
        );
    }
    consume(execution, &clear([0., 0., 1., 1.])).unwrap();
    let pixels = cpu_read(&mut execution.residency, backing);
    assert!(pixels.chunks_exact(4).all(|p| p == [0, 0, 255, 255]));

    let scopes = execution.recording_statistics().rendering_scopes;
    consume(execution, &clear([0., 1., 0., 1.])).unwrap();
    consume(execution, &partial).unwrap();
    assert_eq!(
        execution.recording_statistics().rendering_scopes,
        scopes + 1
    );
    for (index, pixel) in cpu_read(&mut execution.residency, backing)
        .chunks_exact(4)
        .enumerate()
    {
        assert_eq!(
            pixel,
            if index % (area.width as usize) < area.width as usize / 2 {
                &[255, 0, 0, 255]
            } else {
                &[0, 255, 0, 255]
            }
        );
    }

    let (_depth_allocation, depth_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    let mut depth = image.clone();
    depth.description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(8, 8, 1).unwrap(),
        ImageFormat::Depth32Float,
        ImageKind::DepthStencil,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    depth.planes = Arc::from([GuestImagePlane {
        aspect: GuestImageAspect::Depth,
        footprint: crate::residency::tests::plane_geometry(depth_backing.clone(), [8, 8], 32, 4),
        ..depth.planes[0].clone()
    }]);
    let depth_clear = |value| NativeWork {
        frontend: FrontendSubmissionId::new(9002),
        command: NativeCommand::Clear {
            image: depth.clone(),
            rectangle: area,
            value: ClearValue::Depth(value),
        },
    };
    execution.residency.device.end_rendering();
    let scopes = execution.recording_statistics().rendering_scopes;
    consume(execution, &clear([1., 0., 0., 1.])).unwrap();
    consume(execution, &depth_clear(0.375)).unwrap();
    assert_eq!(execution.recording_statistics().rendering_scopes, scopes);
    execution.residency.flush().unwrap();
    // Compatible color/depth clears materialize in one rendering scope.
    assert_eq!(
        execution.recording_statistics().rendering_scopes,
        scopes + 1
    );
    assert!(
        cpu_read(&mut execution.residency, &depth_backing)
            .chunks_exact(4)
            .all(|pixel| f32::from_le_bytes(pixel.try_into().unwrap()) == 0.375)
    );

    let mut read_only = (**state).clone();
    read_only.identity = PreparedStateId::new(9003, 1);
    read_only.depth_stencil = Some(depth.clone());
    let parameters = Arc::make_mut(&mut read_only.parameters);
    parameters.identity = PreparedStateId::new(9004, 1);
    parameters.depth_stencil_format = Some(ImageFormat::Depth32Float);
    parameters.depth = DepthState::DISABLED;
    let scopes = execution.recording_statistics().rendering_scopes;
    consume(execution, &depth_clear(0.625)).unwrap();
    consume(
        execution,
        &NativeWork {
            frontend: FrontendSubmissionId::new(9005),
            command: NativeCommand::Draw {
                state: Arc::new(read_only),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 0,
                    first_instance: 0,
                    instance_count: 1,
                },
            },
        },
    )
    .unwrap();
    // A read-only pass uses STORE_NONE, so its load cannot absorb this clear.
    assert_eq!(
        execution.recording_statistics().rendering_scopes,
        scopes + 2
    );
    assert!(
        cpu_read(&mut execution.residency, &depth_backing)
            .chunks_exact(4)
            .all(|pixel| f32::from_le_bytes(pixel.try_into().unwrap()) == 0.625)
    );
}

fn pipeline_switches_retain_parameters_and_known_transitions(
    execution: &mut NativeGpuExecution,
    a: &Arc<GuestDrawState>,
    b: &Arc<GuestDrawState>,
) {
    assert!(execution.graphics(a).unwrap());
    let first = execution.current_graphics.unwrap().0;
    let retained = Arc::clone(
        &execution.graphics_pipelines[first]
            .parameters
            .as_ref()
            .unwrap()
            .2,
    );
    assert!(execution.graphics(b).unwrap());
    let second = execution.current_graphics.unwrap().0;
    assert_ne!(first, second);
    assert!(execution.graphics(a).unwrap());
    assert!(
        execution.graphics_pipelines[first]
            .transitions
            .contains(&second)
    );
    assert!(
        execution.graphics_pipelines[second]
            .transitions
            .contains(&first)
    );
    let transitions = [first, second].map(|i| execution.graphics_pipelines[i].transitions.len());
    let pipelines = execution.graphics_pipelines.len();
    for _ in 0..4 {
        assert!(execution.graphics(b).unwrap());
        assert_eq!(execution.current_graphics.unwrap().0, second);
        assert!(execution.graphics(a).unwrap());
        assert_eq!(execution.current_graphics.unwrap().0, first);
        assert!(Arc::ptr_eq(
            &retained,
            &execution.graphics_pipelines[first]
                .parameters
                .as_ref()
                .unwrap()
                .2
        ));
    }
    assert_eq!(execution.graphics_pipelines.len(), pipelines);
    assert_eq!(
        [first, second].map(|i| execution.graphics_pipelines[i].transitions.len()),
        transitions
    );
}

fn constant_blend_factors_follow_dynamic_constants(
    execution: &mut NativeGpuExecution,
    template: &Arc<GuestDrawState>,
    output: &Arc<PhysicalMapping>,
) {
    use nixe_gpu::{BlendComponent, BlendFactor as F, BlendOperation, ColorBlendState};
    for (index, factor) in [
        F::ConstantColor,
        F::OneMinusConstantColor,
        F::ConstantAlpha,
        F::OneMinusConstantAlpha,
    ]
    .into_iter()
    .enumerate()
    {
        for (phase, (constants, expected)) in [
            ([0.5, 0.75, 0.5, 0.25], [191, 64, 64, 191]),
            ([0.75, 0.25, 0.75, 0.75], [64, 191, 191, 64]),
        ]
        .into_iter()
        .enumerate()
        {
            let mut draw = (**template).clone();
            draw.identity = PreparedStateId::new(30000 + (index * 2 + phase) as u64, 1);
            let parameters = Arc::make_mut(&mut draw.parameters);
            parameters.identity = draw.identity;
            parameters.shader_state.force_early_z = index % 2 != 0;
            parameters.blend_constants = constants;
            parameters.color_outputs[0].blend = Some(ColorBlendState {
                color: BlendComponent {
                    operation: BlendOperation::Add,
                    source: factor,
                    destination: F::Zero,
                },
                alpha: BlendComponent {
                    operation: BlendOperation::Add,
                    source: F::One,
                    destination: F::Zero,
                },
            });
            let work = NativeWork {
                frontend: FrontendSubmissionId::new(30000 + (index * 2 + phase) as u64),
                command: NativeCommand::Draw {
                    state: Arc::new(draw),
                    arguments: DrawArguments::NonIndexed {
                        first_vertex: 0,
                        vertex_count: 3,
                        first_instance: 0,
                        instance_count: 1,
                    },
                },
            };
            consume(execution, &work)
                .and_then(|()| execution.signal_fence(work.frontend))
                .unwrap();
            for pixel in cpu_read(&mut execution.residency, output).chunks_exact(4) {
                assert_eq!(
                    pixel,
                    [0, expected[index], 0, 255],
                    "factor={factor:?} phase={phase}"
                );
            }
            finish_next(execution).unwrap().unwrap();
        }
    }
}

fn overlapping_graphics_buffers_preserve_shader_bindings(
    execution: &mut NativeGpuExecution,
    template: &Arc<GuestDrawState>,
    output: &Arc<PhysicalMapping>,
) {
    let cache_page = 64 * 1024;
    // Resolve a small uniform first, then grow its physical buffer through a
    // vertex or index range crossing the page boundary. The shader must bind
    // the surviving buffer, including the uniform value written by the GPU.
    // The last case reverses the dependency: a shader range grows the vertex buffer.
    for case in 0..3 {
        let indexed = case == 1;
        let (allocation, mapping) = backing_allocation(
            &execution.residency,
            2 * cache_page as usize,
            cache_page as usize,
            0,
        );
        let uniform_offset = if case == 2 { cache_page - 16 } else { 0 };
        let uniform = mapping
            .slice(uniform_offset, if case == 2 { 32 } else { 16 })
            .unwrap();
        let mut draw = (**template).clone();
        draw.identity = PreparedStateId::new(10000 + case, 1);
        let program = Arc::make_mut(&mut draw.program);
        program.identity = PreparedStateId::new(11000 + case, 1);
        program.bindings = Box::new([(
            uniform_color_resource(),
            GuestBinding::Buffer(Arc::clone(&uniform)),
        )]);
        let arguments = if indexed {
            let offset = cache_page - 2;
            crate::residency::tests::cpu_write(
                &mut execution.residency,
                &allocation,
                offset,
                &[0, 0, 1, 0, 2, 0],
            )
            .unwrap();
            draw.index = Some(GuestIndexBuffer::Canonical(
                mapping.slice(offset, 6).unwrap(),
                IndexType::Uint16,
            ));
            DrawArguments::Indexed {
                first_index: 0,
                index_count: 3,
                vertex_offset: 0,
                first_instance: 0,
                instance_count: 1,
            }
        } else {
            let offset = if case == 2 { 64 } else { cache_page - 24 };
            let vertices: [f32; 18] = [
                -1., -1., 0., 0., 1., 0., 3., -1., 0., 0., 1., 0., -1., 3., 0., 0., 1., 0.,
            ];
            crate::residency::tests::cpu_write(
                &mut execution.residency,
                &allocation,
                offset,
                &vertices
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            draw.vertices = Arc::from([GuestVertexBuffer {
                binding: 0,
                backing: mapping.slice(offset, 72).unwrap(),
            }]);
            draw.index = None;
            DrawArguments::NonIndexed {
                first_vertex: 0,
                vertex_count: 3,
                first_instance: 0,
                instance_count: 1,
            }
        };
        let initial = execution
            .residency
            .buffer_for_deferred_import(
                mapping.slice(0, 16).unwrap(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .unwrap();
        let color = [0f32, 1., 0., 1.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        execution
            .residency
            .write_buffer(initial.buffer, initial.offset + uniform_offset, &color)
            .unwrap();

        let merges = execution.residency.statistics().buffer_merges;
        let work = NativeWork {
            frontend: FrontendSubmissionId::new(10000 + case),
            command: NativeCommand::Draw {
                state: Arc::new(draw),
                arguments,
            },
        };
        for _ in 0..2 {
            consume(execution, &work).unwrap();
            execution.signal_fence(work.frontend).unwrap();
            let bytes = cpu_read(&mut execution.residency, output);
            assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
            finish_next(execution).unwrap().unwrap();
        }
        assert!(execution.residency.statistics().buffer_merges > merges);
        assert!(!execution.residency.buffer_binding_current(initial.buffer));
        assert!(execution.residency.constructing.is_empty());
    }
}

fn contiguous_instances_preserve_vertex_inputs(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
    backing: &PhysicalMapping,
) {
    let (colors, color_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    let values: [f32; 18] = [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
    ];
    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &colors,
        0,
        &values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut draw = (**state).clone();
    draw.identity = PreparedStateId::new(3001, 1);
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(3002, 1);
    parameters.vertices[0].attributes = Box::new([parameters.vertices[0].attributes[0]]);
    let mut layouts = parameters.vertices.to_vec();
    layouts.push(GuestVertexLayout {
        binding: 1,
        stride: 12,
        step: VertexStepMode::Instance,
        attributes: Box::new([nixe_gpu::VertexAttribute {
            shader_location: 1,
            format: VertexFormat::Float32x3,
            offset: 0,
        }]),
    });
    parameters.vertices = layouts.into();
    let mut buffers = draw.vertices.to_vec();
    buffers.push(GuestVertexBuffer {
        binding: 1,
        backing: color_backing.slice(0, 72).unwrap(),
    });
    draw.vertices = buffers.into();
    let draw = Arc::new(draw);
    for grouped in [false, true] {
        let before = execution.recording_statistics().draw_calls;
        let instances = if grouped {
            vec![(3, 3)]
        } else {
            (3..6).map(|id| (id, 1)).collect()
        };
        for (first_instance, instance_count) in instances {
            let submission = NativeWork {
                frontend: FrontendSubmissionId::new(3000),
                command: NativeCommand::Draw {
                    state: Arc::clone(&draw),
                    arguments: DrawArguments::NonIndexed {
                        first_vertex: 0,
                        vertex_count: 3,
                        first_instance,
                        instance_count,
                    },
                },
            };

            consume(execution, &submission)
                .and_then(|()| execution.signal_fence(submission.frontend))
                .unwrap();
        }
        let bytes = cpu_read(&mut execution.residency, backing);
        assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 0, 255, 255]));
        assert_eq!(
            execution.recording_statistics().draw_calls - before,
            if grouped { 1 } else { 3 }
        );
        while finish_next(execution).unwrap().is_some() {}
    }
}

fn window_depth_reconstructs_reads_and_clamps_exports(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
) {
    let (vertices, vertex_backing) = backing_allocation(&execution.residency, 256, 256, 0);
    let data = [
        -1_f32, -1., 0.25, 0., 0., 0., 3., -1., 0.25, 0., 0., 0., -1., 3., 0.25, 0., 0., 0.,
    ];
    crate::residency::tests::cpu_write(
        &mut execution.residency,
        &vertices,
        0,
        &data
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for (index, (transform, explicit)) in [
        (None, false),
        (Some([2., -1.]), false),
        (Some([-2., 1.]), false),
        (Some([4., -1.]), false),
        (Some([2., -1.]), true),
    ]
    .into_iter()
    .enumerate()
    {
        let (pixels, color_backing) = backing_allocation(&execution.residency, 256, 256, 0);
        let (_allocation_3, depth_backing) = backing_allocation(&execution.residency, 256, 256, 0);
        let mut draw = (**state).clone();
        draw.identity = PreparedStateId::new(4000 + index as u64, 1);
        std::sync::Arc::make_mut(&mut draw.vertices)[0].backing =
            vertex_backing.slice(0, 72).unwrap();
        std::sync::Arc::make_mut(&mut draw.colors)[0].planes = Arc::from([GuestImagePlane {
            footprint: crate::residency::tests::plane_geometry(
                color_backing.clone(),
                [8, 8],
                32,
                4,
            ),
            ..draw.colors[0].planes[0].clone()
        }]);
        let mut depth = draw.colors[0].clone();
        depth.description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(8, 8, 1).unwrap(),
            ImageFormat::Depth32Float,
            ImageKind::DepthStencil,
            1,
            1,
            SampleCount::One,
        )
        .unwrap();
        depth.planes = Arc::from([GuestImagePlane {
            aspect: GuestImageAspect::Depth,
            footprint: crate::residency::tests::plane_geometry(
                depth_backing.clone(),
                [8, 8],
                32,
                4,
            ),
            ..depth.planes[0].clone()
        }]);
        draw.depth_stencil = Some(depth);
        let parameters = Arc::make_mut(&mut draw.parameters);
        parameters.identity = PreparedStateId::new(5000 + index as u64, 1);
        parameters.depth_stencil_format = Some(ImageFormat::Depth32Float);
        parameters.depth = DepthState {
            test_enabled: true,
            write_enabled: true,
            compare: DepthCompareOperation::Always,
        };
        let mut viewport = ViewportTransform::new([4., 4., 1.], [4., 4., 0.], [0., 1.])
            .unwrap()
            .with_negative_one_to_one_depth(true);
        if let Some([scale, offset]) = transform {
            viewport = viewport.with_fragment_depth_transform(
                FragmentDepthTransform::new(scale, offset, 0., 1.).unwrap(),
            );
        }
        parameters.viewport = Some(viewport);
        let mut header = [0; 20];
        header[18] = 1;
        header[19] = if explicit { 2 } else { 0 };
        let mut instructions = vec![
            0xe000_0000_0007_ff00 | (30 << 30),
            0x1e00_0000_0007_0000 | (u64::from(0.25_f32.to_bits()) << 20),
            0x0800_0000_0007_0000 | (u64::from(0.5_f32.to_bits()) << 20),
        ];
        if explicit {
            instructions.push(0x0100_0000_0007_f005 | (u64::from(0.75_f32.to_bits()) << 20));
        }
        instructions.push(0xe300_0000_0007_000f);
        let program = Arc::make_mut(&mut draw.program);
        program.identity = PreparedStateId::new(6000 + index as u64, 1);
        program.shader = captured_graphics(
            &execution.device_profile().shader_compiler,
            Some((header, instructions)),
        );
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(4000 + index as u64),
            command: NativeCommand::Draw {
                state: Arc::new(draw),
                arguments: DrawArguments::NonIndexed {
                    first_vertex: 0,
                    vertex_count: 3,
                    first_instance: 0,
                    instance_count: 1,
                },
            },
        };

        consume(execution, &submission)
            .and_then(|()| execution.signal_fence(submission.frontend))
            .unwrap();
        let guest_z: f32 = transform.map_or(0.625, |[scale, offset]| 0.625 * scale + offset);
        let expected_red = ((guest_z * 0.25 + 0.5) * 255.).round() as u8;
        let colors = cpu_read(&mut execution.residency, &color_backing);
        assert!(
            colors
                .chunks_exact(4)
                .all(|p| p[0].abs_diff(expected_red) <= 1),
            "case {index}: {colors:?}"
        );
        let depth = cpu_read(&mut execution.residency, &depth_backing);
        let expected_depth = if explicit {
            0.75
        } else {
            guest_z.clamp(0., 1.)
        };
        for pixel in depth.chunks_exact(4) {
            assert_eq!(
                f32::from_le_bytes(pixel.try_into().unwrap()),
                expected_depth,
                "case {index}"
            );
        }
        drop(pixels);
    }
}

fn render_area_preserves_undefined_padding(
    execution: &mut NativeGpuExecution,
    state: &Arc<GuestDrawState>,
) {
    let (_allocation_4, backing) = backing_allocation(&execution.residency, 256, 256, 0x73);
    let mut draw = (**state).clone();
    draw.identity = PreparedStateId::new(100, 1);
    std::sync::Arc::make_mut(&mut draw.colors)[0].planes = Arc::from([GuestImagePlane {
        footprint: crate::residency::tests::plane_geometry(backing.clone(), [8, 8], 32, 4),
        ..draw.colors[0].planes[0].clone()
    }]);
    std::sync::Arc::make_mut(&mut draw.colors)[0].canonical_texels = false;
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
    let first = NativeWork {
        frontend: FrontendSubmissionId::new(100),
        command: NativeCommand::Sequence(
            vec![
                NativeCommand::Clear {
                    image: image.clone(),
                    rectangle: area,
                    value: ClearValue::Color([0., 0., 1., 1.]),
                },
                NativeCommand::Draw {
                    state: Arc::new(draw.clone()),
                    arguments,
                },
            ]
            .into_boxed_slice(),
        ),
    };
    let parameters = Arc::make_mut(&mut draw.parameters);
    parameters.identity = PreparedStateId::new(102, 1);
    parameters.scissor.x = 3;
    draw.identity = PreparedStateId::new(103, 1);
    let second = NativeWork {
        frontend: FrontendSubmissionId::new(101),
        command: NativeCommand::Draw {
            state: Arc::new(draw),
            arguments,
        },
    };
    let scopes = execution.recording_statistics().rendering_scopes;
    for submission in [&first, &second] {
        consume(execution, submission)
            .and_then(|()| execution.signal_fence(submission.frontend))
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

fn uniform_color_shader(
    configuration: &nixe_gpu::shader_compilation::Configuration,
) -> Arc<GuestShaderCode> {
    let mut header = [0; 20];
    header[18] = 15;
    let instructions = (0..4u64)
        .map(|component| {
            0xef90_0000_0000_0000
                | (4 << 48)
                | ((component * 4) << 20)
                | (7 << 16)
                | (255 << 8)
                | component
        })
        .chain([0xe300_0000_0007_000f])
        .collect();
    captured_graphics(configuration, Some((header, instructions)))
}

fn uniform_color_resource() -> nixe_gpu::shader_module::ResourceId {
    use nixe_shader_recompiler::{
        Stage,
        artifact::{ConstantBufferDescriptor, ResourceSource},
    };
    nixe_gpu::shader_module::guest_resource(
        Stage::Fragment,
        &ResourceSource::ConstantBuffer(ConstantBufferDescriptor::default()),
        0,
        0,
    )
    .unwrap()
}

#[test]
fn native_video_owned_planes_convert_crop_and_reuse_host_images() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
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
        let (_allocation_5, backing) = backing_allocation(&execution.residency, 4096, 4096, 0xa5);
        let backing = backing.slice(0, u64::from(width * height) * bpp).unwrap();
        let image = GuestImage {
            view_identity: None,
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
                depth_slice: 0,
                mip_level: 0,
                array_layer: 0,
                aspect: GuestImageAspect::Color,
                footprint: crate::residency::tests::plane_geometry(
                    backing,
                    [width, height],
                    u64::from(width) * bpp,
                    bpp as u32,
                ),
                row_pitch: u64::from(width) * bpp,
                bytes_per_block: bpp as u32,
            }]),
        };
        (_allocation_5, image)
    };
    let (_output, output) = plane(ImageFormat::Bgra8Unorm, 4, 2);
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
    for (frame, (width, height)) in [(5u32, 3u32), (9, 7), (7, 5), (5, 3), (9, 7)]
        .into_iter()
        .enumerate()
    {
        let uv_width = width.div_ceil(2);
        let ys: Arc<[u8]> = (0..width * height)
            .map(|i| (i * 11 + frame as u32 * 13) as u8)
            .collect();
        let uvs: Arc<[u8]> = (0..uv_width * 2 * height.div_ceil(2))
            .map(|i| (i * 7 + frame as u32 * 9) as u8)
            .collect();
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(frame as u64 + 1),
            command: NativeCommand::ConvertNv12 {
                destination: Box::new(output.clone()),
                source: Nv12Planes {
                    widths: [width, uv_width],
                    bytes: [Arc::clone(&ys), Arc::clone(&uvs)],
                },
                transform,
            },
        };

        consume(&mut execution, &submission)
            .and_then(|()| execution.submit(submission.frontend))
            .unwrap();
        finish_next(&mut execution).unwrap().unwrap();
        let pixels = cpu_read(
            &mut execution.residency,
            output.planes[0].footprint.mapping(),
        );
        for row in 0..2 {
            for column in 0..4 {
                let sx = column + 1;
                let sy = row + 1;
                let values = [
                    i32::from(ys[sy * width as usize + sx]) * 4,
                    i32::from(uvs[(sy / 2 * uv_width as usize + sx / 2) * 2]) * 4,
                    i32::from(uvs[(sy / 2 * uv_width as usize + sx / 2) * 2 + 1]) * 4,
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
        if frame <= 1 {
            descriptor_updates = execution.descriptors.statistics().updates;
        }
        assert_eq!(
            execution.descriptors.statistics().updates,
            descriptor_updates
        );
        assert_eq!(execution.video_pipelines.len(), 1);
        assert_eq!(
            execution.residency.statistics().image_allocations,
            1,
            "decoded host inputs must not register guest image representations"
        );
        assert_eq!(
            execution.video_inputs.as_ref().unwrap().extent,
            if frame == 0 { [5, 3] } else { [9, 7] }
        );
    }
    execution.teardown().unwrap();
    drop(execution);
    monitor.check().unwrap();
}

#[test]
fn captured_compute_dispatch_reuses_prepared_pipeline_without_old_ir() {
    use nixe_gpu::shader_compilation::{ShaderInputs, StageSource};
    use nixe_shader_recompiler::{
        Stage,
        artifact::ShaderInfo,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        types::ProgramOffset,
    };
    let Some(device) = support::device_for_test(ApiPreference::Compatibility12) else {
        return;
    };
    let monitor = device.validation_monitor();
    let inputs = Arc::new(ShaderInputs::new(
        vec![StageSource {
            environment: Arc::new(SnapshotEnvironment::new(
                EnvironmentMetadata {
                    stage: Stage::Compute,
                    workgroup_size: [4, 1, 1],
                    ..Default::default()
                },
                Observations {
                    instructions: [(ProgramOffset(8), 0xe300_0000_0007_000f)].into(),
                    ..Default::default()
                },
            )),
            info: Arc::new(ShaderInfo::default()),
            binding_group: None,
        }]
        .into(),
        &device.profile().shader_compiler,
    ));
    let program = Arc::new(GuestProgram {
        identity: PreparedStateId::new(1, 1),
        shader: inputs,
        bindings: Box::new([]),
        shared_memory_bytes: 0,
    });
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"new-compute-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_allocation, backing) = backing_allocation(&execution.residency, 4096, 4096, 0x19);
    let buffer = execution
        .residency
        .buffer_for_deferred_import(backing.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    execution
        .residency
        .write_buffer(buffer.buffer, buffer.offset, &[0x37; 16])
        .unwrap();
    for value in 1..=2 {
        let batch = NativeWork {
            frontend: FrontendSubmissionId::new(value),
            command: NativeCommand::Dispatch {
                program: Arc::clone(&program),
                groups: [value as u32, 1, 1],
            },
        };

        consume(&mut execution, &batch).unwrap();
        let scheduled = execution.residency.device().scheduled_tick();
        execution.end_command_list().unwrap();
        let sealed = execution.residency.device().scheduled_tick();
        assert!(
            sealed > scheduled,
            "a completed list seals its recorded dispatches"
        );
        execution.end_command_list().unwrap();
        assert_eq!(execution.residency.device().scheduled_tick(), sealed);
        assert!(
            execution.completions.is_empty(),
            "sealing does not create guest completions"
        );
        assert!(
            execution.residency.buffer_visibility_work().is_none(),
            "native submission does not commit unrequested CPU downloads"
        );
        execution.signal_fence(batch.frontend).unwrap();
        finish_next(&mut execution).unwrap().unwrap();
        assert_eq!(
            &cpu_read(&mut execution.residency, &backing)[..16],
            &[0x37; 16]
        );
    }
    assert_eq!(execution.statistics.dispatches, 2);
    assert_eq!(execution.computing.usage().0, 1);
    let source = Arc::downgrade(&program);
    drop(program);
    assert!(source.upgrade().is_none(), "ready compute state is weak");
    execution.computing.prune(16, |computing| match computing {
        Computing::Ready { source, .. } => source.strong_count() != 0,
        Computing::Shader { .. } => true,
    });
    assert_eq!(execution.computing.usage().0, 0);
    drop(execution);
    monitor.check().unwrap();
}

#[test]
fn recorder_prefixes_retain_gpu_versions_without_allocating_completion_records() {
    for read_before_final in [false, true] {
        let Some(device) = support::device_for_test(ApiPreference::Auto) else {
            return;
        };
        let monitor = device.validation_monitor();
        let mut execution = NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            device,
            ShaderNamespace {
                guest: *b"prefix-recording",
                isa: 1,
            },
            CacheConfiguration::default(),
            4 * 1024 * 1024,
        )
        .unwrap();
        let (_allocation_6, backing) = backing_allocation(&execution.residency, 4096, 4096, 0x19);
        let buffer = execution
            .residency
            .buffer_for(backing.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap()
            .buffer;
        execution
            .residency
            .write_buffer(buffer, 0, &[0x37; 16])
            .unwrap();
        for value in 1..=128u8 {
            let prefix = NativeWork {
                frontend: FrontendSubmissionId::new(1),
                command: NativeCommand::ReportMemory {
                    fence: true,
                    target: backing.pin(0, backing.bytes()).unwrap(),
                    offset: 64,
                    bytes: Arc::from([value; 16]),
                },
            };

            consume(&mut execution, &prefix).unwrap();
            assert!(execution.completions.is_empty());
            assert!(execution.poll_completion().unwrap().is_none());
        }
        assert_eq!(execution.statistics.batches, 128);
        assert_eq!(execution.residency.device().scheduled_tick(), 0);
        if read_before_final {
            let bytes = cpu_read(&mut execution.residency, &backing);
            assert_eq!(&bytes[..16], &[0x37; 16]);
            assert_eq!(&bytes[64..80], &[128; 16]);
            assert!(
                execution.completions.is_empty(),
                "CPU visibility must not synthesize a delivery completion"
            );
            assert!(execution.poll_completion().unwrap().is_none());
        }
        let tail = NativeWork {
            frontend: FrontendSubmissionId::new(1),
            command: NativeCommand::WaitForIdle,
        };

        let token = consume(&mut execution, &tail)
            .and_then(|()| execution.submit(tail.frontend))
            .unwrap();
        assert_eq!(execution.completions.len(), 1);
        let completed = finish_next(&mut execution).unwrap().unwrap();
        assert_eq!(completed.submission(), token);
        let bytes = cpu_read(&mut execution.residency, &backing);
        assert_eq!(&bytes[..16], &[0x37; 16]);
        assert_eq!(&bytes[64..80], &[128; 16]);
        execution.teardown().unwrap();
        drop(execution);
        monitor.check().unwrap();
    }
}

fn pressure_execution(budget: u64) -> Option<NativeGpuExecution> {
    Some(
        NativeGpuExecution::new(
            nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
            support::device_for_test(ApiPreference::Auto)?,
            ShaderNamespace {
                guest: *b"nixe-native-test",
                isa: 1,
            },
            CacheConfiguration::default(),
            budget,
        )
        .unwrap(),
    )
}
fn pressure_image(execution: &NativeGpuExecution) -> (nixe_memory::Allocation, GuestImage) {
    let (allocation, backing) = backing_allocation(&execution.residency, 16384, 4096, 0);
    (
        allocation,
        GuestImage {
            view_identity: None,
            description: nixe_gpu::ImageDescription::new(
                ImageDimension::Two,
                nixe_gpu::ImageExtent::new(64, 64, 1).unwrap(),
                ImageFormat::Rgba8Unorm,
                nixe_gpu::ImageKind::Color,
                1,
                1,
                SampleCount::One,
            )
            .unwrap(),
            swizzle: nixe_gpu::Swizzle::IDENTITY,
            canonical_texels: true,
            sample_grid: [1, 1],
            planes: Arc::from([GuestImagePlane {
                depth_slice: 0,
                mip_level: 0,
                array_layer: 0,
                aspect: nixe_gpu::native::GuestImageAspect::Color,
                footprint: crate::residency::tests::plane_geometry(backing, [64, 64], 256, 4),
                row_pitch: 256,
                bytes_per_block: 4,
            }]),
        },
    )
}
#[test]
fn blit_keeps_selected_source_while_allocating_destination() {
    let charged = {
        let Some(mut probe) = pressure_execution(16 * 1024 * 1024) else {
            return;
        };
        let (_allocation, source) = pressure_image(&probe);
        let before = probe.residency.allocated_bytes();
        probe.image(&source).unwrap();
        let charge = probe.residency.allocated_bytes() - before;
        probe.teardown().unwrap();
        charge
    };
    let mut execution = pressure_execution(2 * charged).unwrap();
    let validation = execution.residency.device().validation_monitor();
    let (_a, source) = pressure_image(&execution);
    let (_b, unrelated) = pressure_image(&execution);
    let (_c, destination) = pressure_image(&execution);
    execution.image(&source).unwrap();
    execution.image(&unrelated).unwrap();
    let rect = nixe_gpu::ScissorRect {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let result = execution.blit(&source, &destination, rect, rect, false, false, None);
    assert!(execution.residency.constructing.is_empty());
    assert!(
        execution
            .blit(
                &source,
                &destination,
                nixe_gpu::ScissorRect { x: 64, ..rect },
                rect,
                false,
                false,
                None
            )
            .is_err()
    );
    assert!(
        execution.residency.constructing.is_empty(),
        "failed preparation releases construction retention"
    );
    execution.teardown().unwrap();
    drop(execution);
    validation.check().unwrap();
    assert!(
        result.is_ok(),
        "budget holds both operands after evicting the unrelated image"
    );
}

#[test]
fn observer_proven_gpu_completion_retires_resources_on_the_owner() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let work = device.work_ticket().unwrap();
    let buffer = Arc::new(
        device
            .allocate_buffer(
                1024,
                vk::BufferUsageFlags::TRANSFER_SRC,
                crate::MemoryLocation::GpuOnly,
                crate::AllocationMode::Managed,
                "observer retired buffer",
            )
            .unwrap(),
    );
    let released = Arc::downgrade(&buffer);
    device.retire_buffer(buffer, work).unwrap();
    let (notify, notifications) = std::sync::mpsc::channel();
    let observer = completion::CompletionWakeup::new(
        &device,
        Arc::new(move || {
            let _ = notify.send(());
        }),
    )
    .unwrap();
    observer.arm(work.tick()).unwrap();
    device.flush().unwrap();
    notifications.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(device.submission.completed, 0);
    assert_eq!(device.submission.progress.completed(), work.tick());
    assert!(released.upgrade().is_some());
    assert_eq!(device.poll_work(work).unwrap(), crate::WorkStatus::Complete);
    assert_eq!(device.completed_tick(), work.tick());
    assert!(released.upgrade().is_none());
    drop(observer);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn stream_fences_publish_their_cpu_visible_prefix_before_later_gpu_work() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let owner = Arc::clone(&device.owner);
    let mut execution = NativeGpuExecution::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        ShaderNamespace {
            guest: *b"stream-fences!!!",
            isa: 1,
        },
        CacheConfiguration::default(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let (_first, first) = backing_allocation(&execution.residency, 4096, 4096, 0x19);
    let (_later, later) = backing_allocation(&execution.residency, 4096, 4096, 0x29);
    let first_buffer = execution
        .residency
        .buffer_for_deferred_import(first.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    execution
        .residency
        .write_buffer(first_buffer.buffer, first_buffer.offset, &[0x37; 16])
        .unwrap();
    let frontend = FrontendSubmissionId::new(1);
    let prefix = execution.signal_fence(frontend).unwrap();

    let mut kind =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let gate = unsafe {
        owner.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )
    }
    .unwrap();
    execution.residency.order_device().unwrap();
    execution.residency.device_mut().begin_recording().unwrap();
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
    let later_buffer = execution
        .residency
        .buffer_for_deferred_import(later.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    execution
        .residency
        .write_buffer(later_buffer.buffer, later_buffer.offset, &[0x71; 16])
        .unwrap();
    // A known release payload must follow the real CPU-visible copies of its
    // prefix, even when its buffer was visited before the native producer.
    execution
        .residency
        .report_memory(first.pin(64, 16).unwrap(), 0, &Arc::from([0xa3; 16]), true)
        .unwrap();
    let suffix = execution.signal_fence(frontend).unwrap();
    let reached = execution
        .completion(Some(Duration::from_secs(5)))
        .unwrap()
        .unwrap();
    assert_eq!(reached.submission(), prefix);
    assert!(
        execution.poll_completion().unwrap().is_none(),
        "the later GPU copy is still blocked"
    );
    assert_eq!(
        cpu_read(&mut execution.residency, &first.slice(0, 16).unwrap()),
        [0x37; 16]
    );
    let (request, mut release) = execution
        .residency
        .memory()
        .prepare_device_access(
            first.pin(64, 16).unwrap(),
            nixe_memory::AccessOperation::Read(Vec::new()),
        )
        .unwrap();
    execution.queue_device_access(request).unwrap();
    assert!(
        release.poll().is_none(),
        "the release payload must follow its CPU-visible GPU copies"
    );
    unsafe {
        owner
            .device
            .signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(gate).value(1))
    }
    .unwrap();
    let reached = execution
        .completion(Some(Duration::from_secs(5)))
        .unwrap()
        .unwrap();
    assert_eq!(reached.submission(), suffix);
    assert_eq!(
        &cpu_read(&mut execution.residency, &later)[..16],
        &[0x71; 16]
    );
    assert_eq!(
        &cpu_read(&mut execution.residency, &first)[64..80],
        &[0xa3; 16]
    );
    execution.progress().unwrap();
    assert!(release.poll().is_some_and(
        |result| matches!(result, Ok(nixe_memory::AccessResult::Read(bytes)) if bytes == [0xa3; 16])
    ));
    // No native CPU copies remain; a logical fence need not wait on GPU-only
    // work and must not manufacture a new native submission.
    let tick = execution.residency.device().submitted_tick();
    let stub = execution.signal_fence(frontend).unwrap();
    assert_eq!(execution.residency.device().submitted_tick(), tick);
    assert_eq!(
        execution.poll_completion().unwrap().unwrap().submission(),
        stub
    );
    drop(execution);
    unsafe {
        owner.device.destroy_semaphore(gate, None);
    }
    monitor.check().unwrap();
}
