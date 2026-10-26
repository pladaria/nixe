use ash::vk;
use nixe_gpu::physical::{
    ByteRect, Footprint, ImageFootprint, ImageStorage, PhysicalMapping, WriteDomain,
};
use nixe_gpu_vulkan::{ApiPreference, VulkanResidency};
use nixe_memory::{AllocationRange, Machine, PAGE_SIZE};
use std::sync::Arc;

#[path = "support/device.rs"]
mod support;

fn padded_rgba_image(input: &AllocationRange) -> nixe_gpu::native::GuestImage {
    use nixe_gpu::{
        ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind, SampleCount,
        Swizzle,
        native::{GuestImage, GuestImageAspect, GuestImagePlane},
    };
    let footprint = Footprint::new(
        PhysicalMapping::new(std::slice::from_ref(input)).unwrap(),
        WriteDomain::Image(ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: 4,
            rows: 4,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: ImageStorage::Pitch { row_pitch: 32 },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 16,
            rect: ByteRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
        }),
    )
    .unwrap();
    GuestImage {
        view_identity: None,
        description: ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(4, 4, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
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
            footprint,
            row_pitch: 16,
            bytes_per_block: 4,
        }]),
        swizzle: Swizzle::IDENTITY,
        sample_grid: [1; 2],
        canonical_texels: true,
    }
}

#[test]
fn native_ram_read_resolves_neighbouring_image_and_reopens_both_aliases() {
    use nixe_gpu::{
        ClearValue, FrontendSubmissionId, GpuExecution, ScissorRect,
        native::{NativeCommand, NativeWork},
        shader_identity::ShaderNamespace,
    };
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    use nixe_memory::{
        AccessOperation, AccessResult, AddressSpace, AddressWidth, GuestVirtualAddress,
        MemoryPermissions,
    };
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
        space
            .map(
                GuestVirtualAddress::new(base),
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let core = space.register_core(Arc::new(|| {}));
    let mut execution = NativeGpuExecution::new(
        machine,
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let submission = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::Clear {
            image: padded_rgba_image(&input),
            rectangle: ScissorRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            value: ClearValue::Color([1., 0., 0., 1.]),
        },
    };

    execution.consume(&submission).unwrap();
    let before = execution.residency_statistics().native_readback_bytes;
    let AccessResult::NativeRead {
        bytes,
        area: Some(area),
    } = access(
        &mut execution,
        vec![input.slice(512, 4).unwrap()],
        AccessOperation::ReadNative,
    )
    else {
        panic!("RAM padding must produce a reusable native read area");
    };
    assert_eq!(*bytes, [0; 4]);
    let downloaded = execution.residency_statistics().native_readback_bytes;
    assert!(downloaded > before);
    {
        let native = core.enter_native().unwrap();
        for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
            for _ in 0..1000 {
                assert_eq!(
                    space
                        .read_native(&native, GuestVirtualAddress::new(base + 512), 4, None)
                        .unwrap(),
                    Some(0)
                );
            }
            assert_eq!(
                space
                    .read_native(&native, GuestVirtualAddress::new(base), 4, None)
                    .unwrap(),
                Some(u32::from_le_bytes([255, 0, 0, 255]).into())
            );
        }
    }
    assert_eq!(
        execution.residency_statistics().native_readback_bytes,
        downloaded
    );
    assert!(matches!(
        access(
            &mut execution,
            vec![input.slice(512, 4).unwrap()],
            AccessOperation::Write(Box::new(9u32.to_le_bytes()))
        ),
        AccessResult::Written
    ));
    assert!(!area.observation().is_invalid());
    {
        let native = core.enter_native().unwrap();
        assert_eq!(
            space
                .read_native(
                    &native,
                    GuestVirtualAddress::new(PAGE_SIZE + 512),
                    4,
                    Some(&area)
                )
                .unwrap(),
            Some(9)
        );
    }
    let AccessResult::Read(pixel) = access(
        &mut execution,
        vec![input.slice(0, 4).unwrap()],
        AccessOperation::Read,
    ) else {
        panic!("expected actual image bytes");
    };
    assert_eq!(*pixel, [255, 0, 0, 255]);
    assert_eq!(
        execution.residency_statistics().native_readback_bytes,
        downloaded
    );
    execution.teardown().unwrap();
    drop(execution);
    validation.check().unwrap();
}

#[test]
fn recorded_literal_publishes_after_native_completion_without_a_frontend_token() {
    use nixe_gpu::{
        FrontendSubmissionId, GpuExecution,
        native::{NativeCommand, NativeWork},
        shader_identity::ShaderNamespace,
    };
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    use nixe_memory::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut execution = NativeGpuExecution::new(
        machine.clone(),
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let (wake, notifications) = std::sync::mpsc::channel();
    execution
        .bind_host_wakeup(Arc::new(move || {
            let _ = wake.send(());
        }))
        .unwrap();
    let submission = NativeWork {
        frontend: FrontendSubmissionId::new(1),
        command: NativeCommand::ReportMemory {
            fence: true,
            target: vec![range.clone()],
            offset: 128,
            bytes: Arc::from(0x12345678u32.to_le_bytes()),
        },
    };

    execution.consume(&submission).unwrap();
    // Match the owner loop: progress, arm its wakeup, then park. Neither a
    // final frontend delivery nor a CPU read may be needed to submit a release.
    let mut actual = [0; 4];
    let release = range.slice(128, 4).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        assert!(std::time::Instant::now() < deadline);
        execution.progress().unwrap();
        assert!(execution.poll_completion().unwrap().is_none());
        machine
            .capture_stream(std::slice::from_ref(&release), &mut actual)
            .unwrap();
        if actual == 0x12345678u32.to_le_bytes() {
            break;
        }
        execution.arm_completion_wakeup().unwrap();
        let wake_at = execution.pending_submit_deadline().unwrap_or(deadline);
        match notifications
            .recv_timeout(wake_at.saturating_duration_since(std::time::Instant::now()))
        {
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("native completion wake disconnected")
            }
        }
    }
    assert!(matches!(
        space
            .read_value(GuestVirtualAddress::new(PAGE_SIZE + 128), 4)
            .unwrap(),
        Access::Ready(0x12345678)
    ));
    assert_eq!(execution.residency_statistics().native_readback_bytes, 0);
    execution.teardown().unwrap();
    drop(execution);
    validation.check().unwrap();
}

#[test]
fn ordered_literal_fences_publish_ram_and_reopen_aliases_without_readback() {
    use nixe_gpu::{
        FrontendSubmissionId, GpuExecution,
        native::{NativeCommand, NativeWork},
        shader_identity::ShaderNamespace,
    };
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    use nixe_memory::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
        space
            .map(
                GuestVirtualAddress::new(base),
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let core = space.register_core(Arc::new(|| {}));
    let mut execution = NativeGpuExecution::new(
        machine.clone(),
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    for round in 0..16u8 {
        let before = execution.residency_statistics().native_readback_bytes;
        for (index, value) in [round + 1, round + 21].into_iter().enumerate() {
            let submission = NativeWork {
                frontend: FrontendSubmissionId::new(u64::from(round) * 2 + index as u64 + 1),
                command: NativeCommand::ReportMemory {
                    fence: true,
                    target: vec![range.clone()],
                    offset: 128,
                    bytes: Arc::from([value; 8]),
                },
            };

            execution
                .consume(&submission)
                .and_then(|consumed| {
                    assert!(consumed);
                    execution.submit(submission.frontend)
                })
                .unwrap();
        }
        if round == 0 {
            let mut before_fence = [0; 8];
            machine
                .capture_stream(&[range.slice(128, 8).unwrap()], &mut before_fence)
                .unwrap();
            assert_eq!(
                before_fence, [0; 8],
                "acceptance must not publish a release before its fence"
            );
        }
        // A later CPU overwrite must survive both queued literal publications.
        let native = core.enter_native().unwrap();
        let address = GuestVirtualAddress::new(3 * PAGE_SIZE + 131);
        let written = space.write_native(&native, address, &[99]).unwrap();
        drop(native);
        if !written {
            // A retained buffer still observes CPU writes after device coverage
            // retires. Resolve that first store through the normal owner path.
            let ranges = space.capture(address, 1, MemoryPermissions::WRITE).unwrap();
            assert!(matches!(
                access(
                    &mut execution,
                    ranges,
                    nixe_memory::AccessOperation::Write(Box::new([99]))
                ),
                nixe_memory::AccessResult::Written
            ));
        }
        execution.progress().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        for _ in 0..2 {
            while execution.poll_completion().unwrap().is_none() {
                assert!(std::time::Instant::now() < deadline);
                execution.progress().unwrap();
                std::thread::yield_now();
            }
        }
        let mut expected = [round + 21; 8];
        expected[3] = 99;
        for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
            assert!(matches!(
                space.read_value(GuestVirtualAddress::new(base + 128), 8).unwrap(),
                Access::Ready(value) if value == u128::from(u64::from_le_bytes(expected))
            ));
            assert!(matches!(
                space
                    .read_value(GuestVirtualAddress::new(base + 512), 8)
                    .unwrap(),
                Access::Ready(0)
            ));
        }
        assert_eq!(
            execution.residency_statistics().native_readback_bytes,
            before
        );
    }
    execution.teardown().unwrap();
    drop(execution);
    validation.check().unwrap();
}

#[test]
fn buffer_fence_downloads_publish_current_ranges_before_later_cpu_scalar_reads() {
    use nixe_memory::{Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut residency = VulkanResidency::new(machine.clone(), device, 8 * 1024 * 1024).unwrap();
    let id = residency
        .buffer_for(
            PhysicalMapping::new(std::slice::from_ref(&range)).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .buffer;
    assert_eq!(
        residency
            .write_buffer(id, 0, &vec![7; PAGE_SIZE as usize])
            .unwrap()
            .consumed,
        PAGE_SIZE as usize
    );
    residency.flush().unwrap();
    assert_eq!(
        residency
            .write_buffer(id, 3, &[44, 55, 66])
            .unwrap()
            .consumed,
        3
    );
    residency.flush().unwrap();
    // Wait through an ordinary small native read, then poll publication using
    // the public flush boundary. No CPU request asks to download the full page.
    assert_eq!(residency.read_buffer(id, 0, 4).unwrap(), [7, 7, 7, 44]);
    residency.flush().unwrap();
    let mut actual = vec![0; PAGE_SIZE as usize];
    machine.capture_stream(&[range], &mut actual).unwrap();
    let mut expected = vec![7; PAGE_SIZE as usize];
    expected[3..6].copy_from_slice(&[44, 55, 66]);
    assert_eq!(actual, expected);
    assert!(matches!(
        space
            .read_value(GuestVirtualAddress::new(PAGE_SIZE + 100), 4)
            .unwrap(),
        Access::Ready(0x07070707)
    ));
    drop(residency);
    validation.check().unwrap();
}

#[test]
fn native_owner_consumes_cpu_alias_overwrites_before_real_image_readback_and_replay() {
    use nixe_gpu::{
        ClearValue, FrontendSubmissionId, GpuExecution, ScissorRect,
        native::{NativeCommand, NativeWork},
        shader_identity::ShaderNamespace,
    };
    use nixe_gpu_vulkan::{NativeGpuExecution, cache::CacheConfiguration};
    use nixe_memory::{
        AccessOperation, AccessResult, AddressSpace, AddressWidth, GuestVirtualAddress,
        MemoryPermissions,
    };
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
        space
            .map(
                GuestVirtualAddress::new(base),
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let image = padded_rgba_image(&input);
    let mut execution = NativeGpuExecution::new(
        machine.clone(),
        device,
        ShaderNamespace {
            guest: *b"nixe-native-test",
            isa: 1,
        },
        CacheConfiguration::default(),
        8 * 1024 * 1024,
    )
    .unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let (done, completed) = std::sync::mpsc::channel();
    let cpu = std::thread::spawn(move || {
        let core = space.register_core(Arc::new(|| {}));
        while let Ok(value) = receive.recv() {
            let native = core.enter_native().unwrap();
            for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
                assert!(
                    space
                        .write_native(
                            &native,
                            GuestVirtualAddress::new(base + 3),
                            &[value, value + 1]
                        )
                        .unwrap()
                );
                assert!(
                    space
                        .write_native(&native, GuestVirtualAddress::new(base + 24), &[99])
                        .unwrap()
                );
            }
            drop(native);
            done.send(()).unwrap();
        }
    });
    for frame in 0..32u8 {
        let submission = NativeWork {
            frontend: FrontendSubmissionId::new(u64::from(frame) + 1),
            command: NativeCommand::Clear {
                image: image.clone(),
                rectangle: ScissorRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                value: ClearValue::Color([1., 0., 0., 1.]),
            },
        };

        execution.consume(&submission).unwrap();
        send.send(frame).unwrap();
        completed.recv().unwrap();
        let AccessResult::Read(actual) = access(
            &mut execution,
            vec![input.slice(0, 128).unwrap()],
            AccessOperation::Read,
        ) else {
            panic!("expected physical readback")
        };
        let mut expected = [0; 128];
        for row in expected.chunks_exact_mut(32) {
            for pixel in row[..16].chunks_exact_mut(4) {
                pixel.copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        expected[3..5].copy_from_slice(&[frame, frame + 1]);
        expected[24] = 99;
        assert_eq!(actual.as_ref(), expected, "physical alias round {frame}");
    }
    drop(send);
    cpu.join().unwrap();
    assert!(execution.residency_statistics().native_readback_bytes > 0);
    execution.teardown().unwrap();
    drop(execution);
    validation.check().unwrap();
}

fn access(
    execution: &mut nixe_gpu_vulkan::NativeGpuExecution,
    ranges: Vec<AllocationRange>,
    operation: nixe_memory::AccessOperation,
) -> nixe_memory::AccessResult {
    use nixe_gpu::{GpuExecution, OwnerAccess};
    match execution.access_memory(ranges, operation).unwrap() {
        OwnerAccess::Ready(result) => result,
        OwnerAccess::Pending(mut pending) => {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                execution.progress().unwrap();
                if let Some(result) = pending.poll() {
                    break result.unwrap();
                }
                execution.arm_completion_wakeup().unwrap();
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}
