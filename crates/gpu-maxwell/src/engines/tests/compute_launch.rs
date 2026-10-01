use super::*;

use nixe_memory::CanonicalWriteBatch;

#[test]
fn compute_storage_pointer_uses_pending_canonical_bytes_and_validates_consumed_ranges() {
    storage_launch_case(None);
}

#[cfg(not(target_os = "macos"))]
#[test]
#[ignore = "requires a physical Vulkan GPU"]
fn compute_launch_executes_translated_stg_through_shared_frontend_and_backend() {
    let Some(backend) = crate::shader::hardware::initialize_backend(
        nixe_gpu::BackendInstanceId::new(780),
        nixe_memory::NonCpuDeviceId::new(780),
        nixe_gpu_wgpu::WgpuBackendConfiguration {
            pipeline_cache_directory: None,
            ..Default::default()
        },
    ) else {
        return;
    };
    storage_launch_case(Some(backend.into_runtime()));
}

fn storage_launch_case(mut runtime: Option<Box<dyn nixe_gpu::NeutralBackendRuntime>>) {
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let request = MaxwellMapRequest {
        allocation: MaxwellAllocationId::new(1),
        backing: backing.clone(),
        backing_offset: 0,
        size: 0x1000,
        allocation_alignment: 0x1000,
        page_size: 0,
        kind: 0,
        cacheable: false,
        permissions: MemoryPermissions::READ_WRITE,
        fixed_offset: None,
    };
    let address = address_space.map(request.clone()).unwrap().offset().get();
    let alias = address_space.map(request.clone()).unwrap().offset().get();
    let read_only = address_space
        .map(MaxwellMapRequest {
            permissions: MemoryPermissions::READ,
            ..request
        })
        .unwrap()
        .offset()
        .get();
    for (offset, word) in [
        (0x20, 0x200_u32),
        (0x30, 1),
        (0x34, 0x0001_0001),
        (0x48, 0x0001_0017),
        (0x4c, 0x0001_0001),
        (0x50, 1),
        (0x74, (alias + 0x800) as u32),
        (0x78, (((alias + 0x800) >> 32) as u32) | (8 << 15)),
        (0xb8, 2 << 24),
    ] {
        allocation.write(offset, &word.to_le_bytes()).unwrap();
    }
    // Proven zero relative offset, 64-bit pointer from c[0][0], then one STG.
    let code: [u64; 8] = [
        0,
        0x4c1080000007ff00,
        0x4c1008000017ff01,
        0xeedc2000000700ff,
        0,
        0xe30000000007000f,
        0,
        0,
    ];
    allocation
        .write(
            0x200,
            &code
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut channel = compute_channel();
    dispatch_first(
        &mut channel,
        &packet_on_subchannel(1, 0x02b4 / 4, (address >> 8) as u32),
    )
    .unwrap();
    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(
            1,
            0x1608 / 4,
            &[(address >> 32) as u32, address as u32],
        ),
    )
    .unwrap();
    let dispatch = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)).unwrap();
    let [MaxwellEngineOperation::ComputeLaunch { launch, state }] = dispatch.ordered_operations()
    else {
        panic!()
    };
    let mut writes = CanonicalWriteBatch::new();
    writes
        .stage(&backing, 0x800, &(address + 0xc00).to_le_bytes())
        .unwrap();
    let resolved = resolve_compute_launch(launch, state, &address_space, &writes).unwrap();
    let mut cache = crate::MaxwellLoweringCache::default();
    let first = cache
        .lower_compute(
            &resolved,
            &address_space,
            &writes,
            nixe_gpu::FrontendSubmissionId::new(1),
            vec![],
        )
        .unwrap();
    let second = cache
        .lower_compute(
            &resolved,
            &address_space,
            &writes,
            nixe_gpu::FrontendSubmissionId::new(2),
            vec![nixe_gpu::FrontendSubmissionId::new(1)],
        )
        .unwrap();
    assert!(
        second.resource_creations().is_empty(),
        "warm dispatch must reuse shaders, buffers, descriptors and pipeline"
    );
    assert!(second.resource_invalidations().is_empty());
    if let Some(runtime) = &mut runtime {
        // Commit the same command-processor write before device consumption.
        // Planning itself must not publish guest memory or completion.
        writes.commit().unwrap();
        allocation.write(0xc00, &[0xa5; 8]).unwrap();
        for work in [&first, &second] {
            runtime
                .submit(
                    work.resource_creations(),
                    work.resource_invalidations(),
                    work.submission(),
                )
                .unwrap();
        }
        let mut point = None;
        while let Some(completion) = runtime.wait_for_completion().unwrap() {
            point = Some(completion.visibility());
        }
        let page = &backing.segments()[0];
        let bytes = runtime
            .make_cpu_visible(nixe_memory::CpuVisibilityRequest {
                page: page.page(),
                size: 4096,
                device: nixe_memory::NonCpuDeviceId::new(780),
                visible_at: point.unwrap(),
            })
            .unwrap();
        assert_eq!(&bytes[0xc00..0xc08], &[0, 0, 0, 0, 0xa5, 0xa5, 0xa5, 0xa5]);
        runtime.teardown().unwrap();
        return;
    }
    // deko3d rotates CB0 with each QMD job and invalidates its constant cache
    // at ring wrap. Changing bytes must not create new resource identities.
    // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/gpu_compute.cpp
    let mut ring_tables = Vec::new();
    for frame in 0..384_u64 {
        let slot = frame % 128;
        let cb = alias + 0x800 + slot * 8;
        let mut pending = CanonicalWriteBatch::new();
        pending
            .stage(&backing, 0x74, &(cb as u32).to_le_bytes())
            .unwrap();
        pending
            .stage(
                &backing,
                0x78,
                &(((cb >> 32) as u32) | (8 << 15)).to_le_bytes(),
            )
            .unwrap();
        // Alternate destination pointers on successive laps: constants remain
        // live even when the descriptor table itself is reused.
        pending
            .stage(
                &backing,
                0x800 + slot * 8,
                &(address + 0xc00 + (frame / 128 % 2) * 4).to_le_bytes(),
            )
            .unwrap();
        let resolved = resolve_compute_launch(launch, state, &address_space, &pending).unwrap();
        let work = cache
            .lower_compute(
                &resolved,
                &address_space,
                &pending,
                nixe_gpu::FrontendSubmissionId::new(10 + frame),
                vec![],
            )
            .unwrap();
        assert!(work.resource_invalidations().is_empty());
        let tables = work
            .submission()
            .operations()
            .iter()
            .find_map(|op| match op.command() {
                nixe_gpu::GpuCommand::Dispatch(dispatch) => {
                    Some(dispatch.descriptor_tables.clone())
                }
                _ => None,
            })
            .unwrap();
        if frame < 256 {
            // Two distinct storage destinations require two descriptor sets
            // per slot, but never another shader or pipeline.
            assert!(work.resource_creations().iter().all(|c| matches!(
                c,
                nixe_gpu::BackendResourceCreateInfo::Buffer { .. }
                    | nixe_gpu::BackendResourceCreateInfo::DescriptorTable { .. }
            )));
            ring_tables.push(tables);
        } else {
            assert!(
                work.resource_creations().is_empty(),
                "warm ring slot {slot}"
            );
            assert_eq!(tables, ring_tables[slot as usize]);
        }
    }
    let mut patched = CanonicalWriteBatch::new();
    patched
        .stage(&backing, 0x218, &0xeedc200000070000_u64.to_le_bytes())
        .unwrap();
    patched
        .stage(&backing, 0x800, &(address + 0xc00).to_le_bytes())
        .unwrap();
    let changed = cache
        .lower_compute(
            &resolved,
            &address_space,
            &patched,
            nixe_gpu::FrontendSubmissionId::new(3),
            vec![],
        )
        .unwrap();
    assert!(
        changed
            .resource_creations()
            .iter()
            .any(|c| matches!(c, nixe_gpu::BackendResourceCreateInfo::Shader { .. }))
    );
    assert!(
        changed
            .resource_invalidations()
            .iter()
            .any(|r| matches!(r, nixe_gpu::ResourceDependency::Shader(_)))
    );
    let program = resolved.translate_kernel(&address_space, &writes).unwrap();
    let resources = resolved
        .resolve_resources(&program, &address_space, &writes)
        .unwrap();
    let storage = resources
        .iter()
        .find(|(binding, _)| *binding == 32)
        .unwrap();
    assert_eq!(storage.1.segments().len(), 1);
    assert_eq!(storage.1.segments()[0].backing_offset(), 0xc00);
    assert_eq!(storage.1.segments()[0].size(), 4);
    let mut unchanged = [0xff; 8];
    allocation.read(0x800, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0; 8]);

    // Aliasing through a distinct GPU mapping is still the same canonical RAM.
    let mut overlapping = CanonicalWriteBatch::new();
    overlapping
        .stage(&backing, 0x800, &(address + 0x804).to_le_bytes())
        .unwrap();
    assert!(matches!(
        cache.lower_compute(
            &resolved,
            &address_space,
            &overlapping,
            nixe_gpu::FrontendSubmissionId::new(4),
            vec![],
        ),
        Err(crate::MaxwellLoweringError::BufferBacking(reason))
            if reason.contains("overlapping writable compute bindings")
    ));

    for pointer in [address + 0xc02, 1 << 40, read_only + 0xc00] {
        let mut writes = CanonicalWriteBatch::new();
        writes
            .stage(&backing, 0x800, &pointer.to_le_bytes())
            .unwrap();
        assert!(
            resolved
                .resolve_resources(&program, &address_space, &writes)
                .is_err()
        );
    }
    // Pointer extraction must honor the descriptor size, not merely mapped RAM.
    allocation
        .write(
            0x78,
            &((((alias + 0x800) >> 32) as u32) | (4 << 15)).to_le_bytes(),
        )
        .unwrap();
    let writes = CanonicalWriteBatch::new();
    let resolved = resolve_compute_launch(launch, state, &address_space, &writes).unwrap();
    assert!(matches!(
        resolved.resolve_resources(&program, &address_space, &writes),
        Err(MaxwellComputeLaunchError::Buffer {
            reason: "shader access exceeds its constant buffer",
            ..
        })
    ));
}

#[test]
fn pcas_address_is_shifted_without_truncation_and_does_not_launch_work() {
    let mut channel = compute_channel();
    let graphics = channel.three_d().clone();
    assert_eq!(channel.compute().qmd_address().value(), None);
    for (raw, expected) in [
        (0, 0),
        (0x0004_0830, 0x0408_3000),
        (u32::MAX, 0xffff_ffff00),
    ] {
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, raw)).unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SEND_PCAS_A"
        );
        assert!(dispatch.ordered_operations().is_empty());
        let register = channel.compute().qmd_address();
        assert_eq!(register.raw(), Some(raw));
        assert_eq!(register.value().unwrap().get(), expected);
        assert_eq!(register.source().unwrap().argument(), raw);
        assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
    }
    assert_eq!(channel.three_d(), &graphics);
    assert_eq!(compute_channel().compute().qmd_address().value(), None);
}

#[test]
fn pcas_schedule_captures_address_flags_and_program_state_at_the_trigger() {
    let mut channel = compute_channel();
    dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(1, 0x1608 / 4, &[4, 0x1000]),
    )
    .unwrap();
    for argument in [2, 3] {
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, argument)).unwrap();
        let [MaxwellEngineOperation::ComputeLaunch { launch, state }] =
            dispatch.ordered_operations()
        else {
            panic!("SCHEDULE must emit exactly one launch");
        };
        assert_eq!(launch.address().get(), 0x0408_3000);
        assert_eq!(launch.invalidate(), argument == 3);
        assert_eq!(launch.source().method(), GpuMethodId(0x02bc));
        assert_eq!(launch.source().argument(), argument);
        assert_eq!(
            state.as_ref().program().region_address().unwrap().get(),
            0x0400001000
        );
        assert_eq!(state.as_ref(), channel.compute());
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x12345)).unwrap();
        assert_eq!(launch.address().get(), 0x0408_3000);
        assert_eq!(state.as_ref().qmd_address().raw(), Some(0x40830));
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    }
}

#[test]
fn pcas_rejects_missing_address_reserved_flags_and_unscheduled_operations() {
    let mut channel = compute_channel();
    let before = channel.clone();
    assert!(matches!(
        dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)),
        Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
            reason: "SCHEDULE requires SEND_PCAS_A",
            ..
        })
    ));
    assert_eq!(channel, before);
    dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02b4 / 4, 0x40830)).unwrap();
    for argument in [0, 1, 4, 0x8000_0003] {
        let before = channel.clone();
        let result = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, argument));
        if argument < 2 {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                    reason: "PCAS without SCHEDULE is not implemented",
                    ..
                })
            ));
        } else {
            assert!(matches!(
                result,
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    defined_mask: 3,
                    ..
                })
            ));
        }
        assert_eq!(channel, before);
    }
}

#[test]
fn qmd_resolution_checks_memory_version_and_program_address_on_consumption() {
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let request = MaxwellMapRequest {
        allocation: MaxwellAllocationId::new(1),
        backing: allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        backing_offset: 0,
        size: 0x1000,
        allocation_alignment: 0x1000,
        page_size: 0,
        kind: 0,
        cacheable: false,
        permissions: MemoryPermissions::READ_WRITE,
        fixed_offset: None,
    };
    let address = address_space.map(request.clone()).unwrap().offset().get();
    let write_only = address_space
        .map(MaxwellMapRequest {
            permissions: MemoryPermissions::WRITE,
            ..request
        })
        .unwrap()
        .offset()
        .get();
    let mut channel = compute_channel();
    let writes = CanonicalWriteBatch::new();

    for qmd_address in [0xffff_ffff00, write_only] {
        // Programming a pointer is legal; validate memory only at scheduling.
        dispatch_first(
            &mut channel,
            &packet_on_subchannel(1, 0x02b4 / 4, (qmd_address >> 8) as u32),
        )
        .unwrap();
        let dispatch =
            dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)).unwrap();
        let [MaxwellEngineOperation::ComputeLaunch { launch, state }] =
            dispatch.ordered_operations()
        else {
            panic!()
        };
        assert!(
            matches!(resolve_compute_launch(launch, state, &address_space, &writes),
            Err(MaxwellComputeLaunchError::Address { address, .. }) if address == qmd_address)
        );
    }

    dispatch_first(
        &mut channel,
        &packet_on_subchannel(1, 0x02b4 / 4, (address >> 8) as u32),
    )
    .unwrap();
    let dispatch = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 3)).unwrap();
    let [MaxwellEngineOperation::ComputeLaunch { launch, state }] = dispatch.ordered_operations()
    else {
        panic!()
    };
    assert!(matches!(
        resolve_compute_launch(launch, state, &address_space, &writes),
        Err(MaxwellComputeLaunchError::QmdVersion {
            major: 0,
            minor: 0,
            ..
        })
    ));

    allocation
        .write(0x48, &0x0020_0017_u32.to_le_bytes())
        .unwrap();
    assert!(matches!(
        resolve_compute_launch(launch, state, &address_space, &writes),
        Err(MaxwellComputeLaunchError::MissingProgramRegion { .. })
    ));

    dispatch_first(
        &mut channel,
        &incrementing_packet_on_subchannel(1, 0x1608 / 4, &[0xff, 0xffff_ff00]),
    )
    .unwrap();
    allocation.write(0x20, &0x100_u32.to_le_bytes()).unwrap();
    let dispatch = dispatch_first(&mut channel, &packet_on_subchannel(1, 0x02bc / 4, 2)).unwrap();
    let [MaxwellEngineOperation::ComputeLaunch { launch, state }] = dispatch.ordered_operations()
    else {
        panic!()
    };
    assert!(matches!(
        resolve_compute_launch(launch, state, &address_space, &writes),
        Err(MaxwellComputeLaunchError::ProgramAddressOverflow {
            base: 0xffff_ffff00,
            offset: 0x100,
            ..
        })
    ));

    // No descriptor cache: even without INVALIDATE, subsequent reads observe
    // modified memory. The last byte address is representable; wrapping is not.
    allocation.write(0x20, &0xff_u32.to_le_bytes()).unwrap();
    let resolved = resolve_compute_launch(launch, state, &address_space, &writes).unwrap();
    assert_eq!(resolved.kernel_key().0, 0xffff_ffffff);
}
