use ash::vk;
use nixe_gpu::physical::{
    ByteRect, Footprint, ImageFootprint, ImageStorage, PhysicalMapping, WriteDomain,
};
use nixe_gpu_vulkan::{ApiPreference, ImagePlane, NativeImageDescription, VulkanResidency};
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
        ClearValue, ExecutionBatch, FrontendSubmissionId, FrontendSubmissionSegment, GpuExecution,
        ScissorRect,
        native::{NativeCommand, NativeSubmission},
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
    let submission = NativeSubmission::new(
        FrontendSubmissionId::new(1),
        FrontendSubmissionSegment::FIRST,
        true,
        vec![NativeCommand::Clear {
            image: padded_rgba_image(&input),
            rectangle: ScissorRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            value: ClearValue::Color([1., 0., 0., 1.]),
        }],
    );
    assert!(
        execution
            .prepare(ExecutionBatch {
                commands: &submission
            })
            .unwrap()
    );
    execution
        .record(ExecutionBatch {
            commands: &submission,
        })
        .unwrap();
    let before = execution.residency_statistics().native_readback_bytes;
    let AccessResult::NativeRead {
        bytes,
        area: Some(area),
    } = execution
        .access_memory(
            vec![input.slice(512, 4).unwrap()],
            AccessOperation::ReadNative,
        )
        .unwrap()
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
        execution
            .access_memory(
                vec![input.slice(512, 4).unwrap()],
                AccessOperation::Write(Box::new(9u32.to_le_bytes()))
            )
            .unwrap(),
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
    let AccessResult::Read(pixel) = execution
        .access_memory(vec![input.slice(0, 4).unwrap()], AccessOperation::Read)
        .unwrap()
    else {
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
        ExecutionBatch, FrontendSubmissionId, FrontendSubmissionSegment, GpuExecution,
        native::{NativeCommand, NativeSubmission},
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
    let submission = NativeSubmission::new(
        FrontendSubmissionId::new(1),
        FrontendSubmissionSegment::FIRST,
        false,
        vec![NativeCommand::ReportMemory {
            fence: true,
            target: vec![range.clone()],
            offset: 128,
            bytes: Arc::from(0x12345678u32.to_le_bytes()),
        }],
    );
    assert!(
        execution
            .prepare(ExecutionBatch {
                commands: &submission
            })
            .unwrap()
    );
    execution
        .record(ExecutionBatch {
            commands: &submission,
        })
        .unwrap();
    // Match the owner loop: progress, arm its wakeup, then park. Neither a
    // final frontend delivery nor a CPU read may be needed to submit a release.
    let mut actual = [0; 4];
    let release = range.slice(128, 4).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        assert!(std::time::Instant::now() < deadline);
        execution.progress_device_accesses().unwrap();
        assert!(execution.poll_completion().unwrap().is_none());
        machine
            .capture_stream(std::slice::from_ref(&release), &mut actual)
            .unwrap();
        if actual == 0x12345678u32.to_le_bytes() {
            break;
        }
        execution.request_completion_wakeup().unwrap();
        notifications
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("recorded release must wake its owner");
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
        ExecutionBatch, FrontendSubmissionId, FrontendSubmissionSegment, GpuExecution,
        native::{NativeCommand, NativeSubmission},
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
            let submission = NativeSubmission::new(
                FrontendSubmissionId::new(u64::from(round) * 2 + index as u64 + 1),
                FrontendSubmissionSegment::FIRST,
                true,
                vec![NativeCommand::ReportMemory {
                    fence: true,
                    target: vec![range.clone()],
                    offset: 128,
                    bytes: Arc::from([value; 8]),
                }],
            );
            assert!(
                execution
                    .prepare(ExecutionBatch {
                        commands: &submission
                    })
                    .unwrap()
            );
            execution
                .submit(ExecutionBatch {
                    commands: &submission,
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
        assert!(
            space
                .write_native(
                    &native,
                    GuestVirtualAddress::new(3 * PAGE_SIZE + 131),
                    &[99]
                )
                .unwrap()
        );
        drop(native);
        execution.synchronize_command_sources().unwrap();
        for _ in 0..2 {
            assert!(execution.poll_completion().unwrap().is_some());
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
        .unwrap();
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
        ClearValue, ExecutionBatch, FrontendSubmissionId, FrontendSubmissionSegment, GpuExecution,
        ScissorRect,
        native::{NativeCommand, NativeSubmission},
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
        let submission = NativeSubmission::new(
            FrontendSubmissionId::new(u64::from(frame) + 1),
            FrontendSubmissionSegment::FIRST,
            true,
            vec![NativeCommand::Clear {
                image: image.clone(),
                rectangle: ScissorRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                value: ClearValue::Color([1., 0., 0., 1.]),
            }],
        );
        assert!(
            execution
                .prepare(ExecutionBatch {
                    commands: &submission
                })
                .unwrap()
        );
        execution
            .record(ExecutionBatch {
                commands: &submission,
            })
            .unwrap();
        send.send(frame).unwrap();
        completed.recv().unwrap();
        let AccessResult::Read(actual) = execution
            .access_memory(vec![input.slice(0, 128).unwrap()], AccessOperation::Read)
            .unwrap()
        else {
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

#[test]
fn cropped_native_aliases_copy_without_a_linear_buffer_representation() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(&[input]).unwrap();
    let image = |width, height, base| {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let geometry = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Image(ImageFootprint {
                base,
                columns: width,
                rows: height,
                packed_bytes: 4,
                aspect_offset: 0,
                aspect_bytes: 4,
                storage: ImageStorage::Pitch { row_pitch: 80 },
                native_base: 0,
                native_element_bytes: 4,
                native_row_pitch: u64::from(width) * 4,
                rect: ByteRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
            }),
        )
        .unwrap();
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UINT,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        };
        let plane = ImagePlane {
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
        };
        (vec![geometry], description, vec![plane])
    };
    let mut residency = VulkanResidency::new(machine.clone(), device, 32 * 1024 * 1024).unwrap();
    let (geometry, description, planes) = image(17, 13, 0);
    let source = residency
        .image_for_device_write(geometry, description, planes)
        .unwrap();
    let (geometry, description, planes) = image(13, 9, 0);
    let target = residency
        .image_for_deferred_import(geometry, description, planes)
        .unwrap();
    let allocated = residency.allocated_bytes();
    for color in [[12, 34, 56, 78], [87, 65, 43, 21]] {
        residency
            .clear_color_image(source, 0, vk::ClearColorValue { uint32: color })
            .unwrap();
        residency.ensure_image(target).unwrap();
        assert_eq!(residency.allocated_bytes(), allocated);
    }
    assert_eq!(machine.tracking_statistics().captured_bytes, 0);
    // Publish a distinct destination from the cropped native image, then read
    // its actual GPU bytes rather than its source authority's representation.
    let (geometry, description, planes) = image(13, 9, PAGE_SIZE);
    let destination = residency
        .image_for_device_write(geometry, description, planes)
        .unwrap();
    residency
        .copy_image_region(
            target,
            0,
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 13,
                    height: 9,
                },
            },
            destination,
            0,
            vk::Offset2D::default(),
        )
        .unwrap();
    residency.preserve_image(destination).unwrap();
    let output = AllocationRange::new(allocation.pin(), PAGE_SIZE, 9 * 80).unwrap();
    let mut bytes = vec![0; 9 * 80];
    machine.capture_stream(&[output], &mut bytes).unwrap();
    for row in bytes.chunks_exact(80) {
        for pixel in row[..13 * 4].chunks_exact(4) {
            assert_eq!(pixel, [87, 65, 43, 21]);
        }
        assert!(row[13 * 4..].iter().all(|&byte| byte == 0));
    }
    drop(residency);
    validation.check().unwrap();
}

#[test]
fn separate_depth_and_stencil_writes_preserve_native_contents_without_ram_reconstruction() {
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
    let input = AllocationRange::new(allocation.pin(), 0, 256).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&input)).unwrap();
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::D24_UNORM_S8_UINT,
        extent: vk::Extent3D {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        flags: vk::ImageCreateFlags::empty(),
    };
    let mut planes = Vec::new();
    let mut geometry = Vec::new();
    for (aspect, offset, bytes, native_bytes, native_base) in [
        (vk::ImageAspectFlags::DEPTH, 1, 3, 4, 0),
        (vk::ImageAspectFlags::STENCIL, 0, 1, 1, 256),
    ] {
        geometry.push(
            Footprint::new(
                Arc::clone(&mapping),
                WriteDomain::Image(ImageFootprint {
                    base: 0,
                    columns: 8,
                    rows: 8,
                    packed_bytes: 4,
                    aspect_offset: offset,
                    aspect_bytes: bytes,
                    storage: ImageStorage::Pitch { row_pitch: 32 },
                    native_base,
                    native_element_bytes: native_bytes,
                    native_row_pitch: 8 * u64::from(native_bytes),
                    rect: ByteRect {
                        x: 0,
                        y: 0,
                        width: 8,
                        height: 8,
                    },
                }),
            )
            .unwrap(),
        );
        planes.push(ImagePlane {
            offset: native_base,
            row_pitch: 8 * u64::from(native_bytes),
            mip_level: 0,
            array_layer: 0,
            aspect,
            extent: description.extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: native_bytes,
        });
    }
    let mut residency = VulkanResidency::new(machine.clone(), device, 32 * 1024 * 1024).unwrap();
    let image = residency.image_for(geometry, description, planes).unwrap();
    let before = machine.tracking_statistics().captured_bytes;
    for stencil in [0x37, 0x5a] {
        residency.clear_depth_stencil(image, 0, 1.0, 0).unwrap();
        residency
            .clear_depth_stencil(image, 1, 0.0, stencil)
            .unwrap();
        residency.ensure_image(image).unwrap();
    }
    assert_eq!(
        machine.tracking_statistics().captured_bytes,
        before,
        "native aspect writes must not recapture their old RAM inputs"
    );
    residency.preserve_image(image).unwrap();
    let mut result = [0; 256];
    machine.capture_stream(&[input], &mut result).unwrap();
    for pixel in result.chunks_exact(4) {
        assert_eq!(pixel, [0x5a, 0xff, 0xff, 0xff]);
    }
    drop(residency);
    validation.check().unwrap();
}
