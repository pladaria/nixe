//! Measurements of the descriptor and retained-binding production paths.
use super::*;
use crate::engines::tests::register_benchmark::ALLOCATIONS;
use crate::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellGpuAddressSpace,
    MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
};

#[test]
#[cfg(target_os = "linux")]
#[ignore = "opt-in matched descriptor CPU/allocation measurement"]
fn descriptor_lookup_throughput() {
    let machine = crate::test_memory::machine();
    let allocation = crate::test_memory::allocate(&machine, 8192);
    let mut space = MaxwellGpuAddressSpace::new(
        machine.clone(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(1),
            backing: crate::test_memory::ranges(&allocation),
            backing_offset: 0,
            size: 8192,
            allocation_alignment: 4096,
            page_size: 0,
            kind: 0,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    let mut cache = MaxwellThreeDRetainedBackingCache::default();
    cache.set_view_limit(4096);
    let mut memory_access = crate::test_memory::access(space.memory().machine());
    let mut builder = ResourceBuilder::new(space.memory(), None, &mut cache, &mut memory_access);
    let address = mapping.offset().get();
    builder
        .descriptor_pool(
            MaxwellThreeDResourceRole::TextureHeaders,
            Some(address),
            Some(255),
        )
        .unwrap();
    for _ in 0..100 {
        builder
            .descriptor_bytes(MaxwellThreeDResourceRole::TextureHeaders, 0)
            .unwrap();
        builder.descriptor_reads.clear();
    }
    let iterations = std::env::var("NIXE_DESCRIPTOR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100000);
    let cpu_ns = || -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let before = cpu_ns();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(
            builder
                .descriptor_bytes(MaxwellThreeDResourceRole::TextureHeaders, 0)
                .unwrap(),
        );
        builder.descriptor_reads.clear();
    }
    let seconds = start.elapsed().as_secs_f64();
    let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    let cpu_nanoseconds = cpu_ns() - before;
    println!(
        "{{\"workload\":\"unchanged_descriptor_consumer\",\"iterations\":{iterations},\"seconds\":{seconds},\"cpu_nanoseconds\":{cpu_nanoseconds},\"allocations\":{allocations}}}"
    );
}

#[test]
#[ignore = "opt-in retained binding cold/warm allocation measurement"]
fn retained_bindings_throughput() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 64),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let roles = [MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 }];
    let mut access = crate::test_memory::access(space.memory().machine());
    for cold in [true, false] {
        let iterations = if cold { 1000 } else { 100000 };
        let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
        let first = cache
            .select(&roles, &[])
            .resolve(channel.three_d(), space.memory(), 4096, &mut access)
            .unwrap()
            .clone();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            if cold {
                cache = MaxwellThreeDResolvedResourceCache::new(1);
            }
            let bindings = cache
                .select(&roles, &[])
                .resolve(channel.three_d(), space.memory(), 4096, &mut access)
                .unwrap();
            if !cold {
                assert_eq!(&first, bindings);
            }
            std::hint::black_box(bindings);
        }
        let elapsed = start.elapsed();
        let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        if !cold {
            assert_eq!(allocations, 0);
        }
        println!(
            "retained bindings: cold={cold}, iterations={iterations}, elapsed={elapsed:?}, allocations={allocations}"
        );
    }
}

#[test]
fn unused_slots_and_inactive_groups_do_not_allocate_or_rebuild() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 64),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let roles = [MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 }];
    let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
    let mut access = crate::test_memory::access(space.memory().machine());
    let first = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap()
        .clone();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let repeated = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    let warm = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    let repeated = repeated.clone();
    assert!((first == repeated));
    program_three_d(&mut channel, 0x2410, 17);
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let unused = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    let unused_count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    let unused = unused.clone();
    assert_eq!(
        first.resource(roles[0]).unwrap().source(),
        unused.resource(roles[0]).unwrap().source()
    );
    cache
        .select(&[], &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    program_three_d(&mut channel, 0x2410, 16);
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let after = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    let after_count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    let after = after.clone();
    assert_eq!((warm, unused_count), (0, 0));
    assert!((first == unused));
    assert!((unused == after));
    println!(
        "AUDIT warm={warm}; unused_slot_rebind={unused_count} same_result={}; after_nonconsuming_operation={after_count} same_result={}",
        (first == unused),
        (unused == after)
    );
}

#[test]
fn changed_binding_preserves_other_slots_and_cache_pressure_preserves_selected_work() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 64),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
        (0x2410, 17),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let roles = [
        MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 },
        MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 1 },
    ];
    let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
    let mut access = crate::test_memory::access(space.memory().machine());
    let first = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap()
        .clone();
    program_three_d(&mut channel, 0x2388, address as u32 + 64);
    program_three_d(&mut channel, 0x2410, 17);
    let changed = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap()
        .clone();
    assert!(first.resource(roles[0]) == changed.resource(roles[0]));
    assert!(first.resource(roles[1]) != changed.resource(roles[1]));
    assert_eq!(
        changed.resource(roles[1]).unwrap().source().offset().get(),
        address + 64
    );
}

#[test]
fn paired_texture_identity_tracks_only_both_consumed_constant_slots() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (_allocation, _space, address) = crate::shader::test_support::mapped_memory();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 64),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 17),
        (0x2410, 49),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let texture =
        MaxwellThreeDTextureReference::expression(0, (1, 0, 0), Some((3, 4, 20))).unwrap();
    let role = MaxwellThreeDResourceRole::Sampler(texture);
    let original = channel.three_d().resource_state_identity(role);
    program_three_d(&mut channel, 0x2388, address as u32 + 64);
    program_three_d(&mut channel, 0x2410, 33);
    assert!(original.matches(channel.three_d()));
    program_three_d(&mut channel, 0x2410, 49);
    assert!(!original.matches(channel.three_d()));
    let changed = channel.three_d().resource_state_identity(role);
    program_three_d(&mut channel, 0x2410, 17);
    assert!(!changed.matches(channel.three_d()));
}

#[test]
fn storage_descriptor_rebinding_reads_once_without_allocations() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 0x200),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let role = MaxwellThreeDResourceRole::StorageBuffer {
        group: 0,
        slot: 0,
        byte_offset: 0x110,
    };
    let sizes = [(role, MaxwellShaderBufferSize::Declared)];
    let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
    let mut access = crate::test_memory::access(space.memory().machine());
    let descriptor = |offset: u64| {
        let mut bytes = (address + offset).to_le_bytes().to_vec();
        bytes.extend(128u32.to_le_bytes());
        bytes.extend([0; 4]);
        bytes
    };
    for offset in [0x400, 0x500] {
        crate::test_memory::write(&allocation, 0x110, &descriptor(offset)).unwrap();
        cache
            .select(&[role], &sizes)
            .resolve(channel.three_d(), space.memory(), 4096, &mut access)
            .unwrap();
    }
    for (label, offset) in [
        ("unchanged", 0x500),
        ("changed", 0x400),
        ("changed_back", 0x500),
    ] {
        crate::test_memory::write(&allocation, 0x110, &descriptor(offset)).unwrap();
        let before = space
            .memory()
            .machine()
            .tracking_statistics()
            .captured_bytes;
        ALLOCATIONS.with(|n| n.set(Some(0)));
        cache
            .select(&[role], &sizes)
            .resolve(channel.three_d(), space.memory(), 4096, &mut access)
            .unwrap();
        let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        assert_eq!(allocations, 0, "warm rebind {label}");
        let copied = space
            .memory()
            .machine()
            .tracking_statistics()
            .captured_bytes
            - before;
        println!("AUDIT storage={label} descriptor_bytes={copied} allocations={allocations}");
        if cfg!(feature = "performance-counters") {
            assert_eq!(copied, 16);
        }
    }
}

#[test]
fn identical_views_share_storage_across_binding_slots() {
    let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut cache = MaxwellThreeDRetainedBackingCache::default();
    cache.set_view_limit(4096);
    let mut access = crate::test_memory::access(space.memory().machine());
    for slot in 0..8 {
        let mut builder = ResourceBuilder::new(space.memory(), None, &mut cache, &mut access);
        builder
            .buffer(
                MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot },
                MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32),
                64,
            )
            .unwrap();
    }
    println!(
        "AUDIT identical_buffer_view_bindings=8 retained_view_entries={}",
        cache.view_entries.len()
    );
    assert_eq!(cache.view_entries.len(), 1);
}

#[test]
fn shared_attachment_views_keep_read_and_write_permissions_distinct() {
    let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
    let mut cache = MaxwellThreeDRetainedBackingCache::default();
    cache.set_view_limit(4096);
    let mut access = crate::test_memory::access(space.memory().machine());
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(4, 4, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let mut bindings = Vec::new();
    for role in [
        MaxwellThreeDResourceRole::BlitSource,
        MaxwellThreeDResourceRole::BlitDestination,
        MaxwellThreeDResourceRole::ColorTarget(0),
    ] {
        let mut builder = ResourceBuilder::new(
            space.memory(),
            Some(MaxwellThreeDSampleMode::Samples1x1),
            &mut cache,
            &mut access,
        );
        builder
            .image(
                role,
                MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32),
                description,
                MaxwellThreeDImageLayout::PitchLinear,
                None,
                0,
                MaxwellThreeDGuestImageFormat::Canonical(ImageFormat::Rgba8Unorm),
                false,
                None,
            )
            .unwrap();
        let MaxwellThreeDResolvedResource::Image { role: actual, view } =
            builder.resources.pop().unwrap()
        else {
            panic!("image");
        };
        assert_eq!(actual, role);
        bindings.push(view);
    }
    assert_eq!(cache.view_entries.len(), 2);
    assert_eq!(bindings[0].access(), MaxwellThreeDResourceAccess::Read);
    assert_eq!(bindings[1].access(), MaxwellThreeDResourceAccess::Write);
    assert!(!Arc::ptr_eq(&bindings[0], &bindings[1]));
    assert!(Arc::ptr_eq(&bindings[1], &bindings[2]));
}

#[test]
fn combined_texture_and_sampler_read_one_handle_per_draw() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (allocation, mut space, address) = crate::shader::test_support::mapped_memory();
    let image = address + 0x400;
    let tic = [
        0x74e2_4915u32,
        image as u32,
        (image >> 32) as u32 | (2 << 21),
        2,
        18 | (1 << 23),
        16 | (1 << 31),
        0,
        0,
    ];
    let tsc = [
        0u32,
        2 | (2 << 4) | (3 << 6),
        (15 * 256) << 12,
        0,
        0,
        0,
        0,
        0,
    ];
    for (at, words) in [(0x100, tic), (0x140, tsc)] {
        crate::test_memory::write(
            &allocation,
            at,
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 128),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
        (0x1234, 0),
        (0x1574, (address >> 32) as u32),
        (0x1578, address as u32 + 0x100),
        (0x157c, 0),
        (0x155c, (address >> 32) as u32),
        (0x1560, address as u32 + 0x140),
        (0x1564, 0),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let texture = MaxwellThreeDTextureReference::new(0, 0, 0);
    let roles = [MaxwellThreeDResourceRole::SampledImage {
        texture,
        dimension: MaxwellThreeDTextureDimension::Two,
        with_sampler: true,
    }];
    let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
    let mut access = crate::test_memory::access(space.memory().machine());
    cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    let before = space
        .memory()
        .machine()
        .tracking_statistics()
        .captured_bytes;
    cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    let read = space
        .memory()
        .machine()
        .tracking_statistics()
        .captured_bytes
        - before;
    println!(
        "AUDIT FOLLOWUP combined_texture_sampler warm_bytes={read}; distinct_handle_bytes=4 TIC=32 TSC=32"
    );
    #[cfg(feature = "performance-counters")]
    assert_eq!(read, 68);
    let first = *cache
        .selected
        .sampler(MaxwellThreeDResourceRole::Sampler(texture))
        .unwrap();
    // A mutable TSC descriptor must be consumed again even with the same handle.
    crate::test_memory::write(&allocation, 0x144, &((tsc[1] & !0xf) | 1).to_le_bytes()).unwrap();
    let selected = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    assert_ne!(
        *selected
            .sampler(MaxwellThreeDResourceRole::Sampler(texture))
            .unwrap(),
        first
    );
    // Then switch the handle to another TSC entry without touching the registers.
    crate::test_memory::write(
        &allocation,
        0x160,
        &tsc.into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    program_three_d(&mut channel, 0x1564, 1);
    crate::test_memory::write(&allocation, 0, &(1u32 << 20).to_le_bytes()).unwrap();
    let selected = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 4096, &mut access)
        .unwrap();
    assert_eq!(
        *selected
            .sampler(MaxwellThreeDResourceRole::Sampler(texture))
            .unwrap(),
        first
    );
    // Handle and descriptor inputs are consumed bytes, not retained resources.
    assert_eq!(cache.retained_backings.entries.usage().0, 1);
    let original = match cache.selected.resource(roles[0]).unwrap() {
        MaxwellThreeDResolvedResource::Image { view, .. } => Arc::clone(view),
        _ => panic!("sampled image"),
    };
    crate::test_memory::write(
        &allocation,
        0x120,
        &tic.into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    program_three_d(&mut channel, 0x157c, 1);
    crate::test_memory::write(&allocation, 0, &(1 | (1u32 << 20)).to_le_bytes()).unwrap();
    // A second TIC index shares the interned view. Changing the TIC invalidates
    // that index, and restoring its bytes recovers the shared view.
    for changed in [false, true, false] {
        crate::test_memory::write(
            &allocation,
            0x124,
            &(tic[1] + if changed { 0x100 } else { 0 }).to_le_bytes(),
        )
        .unwrap();
        let selected = cache
            .select(&roles, &[])
            .resolve(channel.three_d(), space.memory(), 4096, &mut access)
            .unwrap();
        let MaxwellThreeDResolvedResource::Image { view, .. } =
            selected.resource(roles[0]).unwrap()
        else {
            panic!("sampled image")
        };
        assert_eq!(Arc::ptr_eq(&original, view), !changed);
    }
    let weak = Arc::downgrade(&original);
    drop(original);
    cache.retained_backings.set_view_limit(1);
    cache
        .select(
            &[MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 }],
            &[],
        )
        .resolve(channel.three_d(), space.memory(), 1, &mut access)
        .unwrap();
    assert!(
        weak.upgrade().is_none(),
        "descriptor indices must not pin evicted views"
    );
    let selected = cache
        .select(&roles, &[])
        .resolve(channel.three_d(), space.memory(), 1, &mut access)
        .unwrap();
    let MaxwellThreeDResolvedResource::Image { view, .. } = selected.resource(roles[0]).unwrap()
    else {
        panic!("sampled image")
    };
    assert_eq!(view.source().offset().get(), image);
    // A remap can be ordered after interpretation but before caching the result.
    // Reusing that result must validate the generation captured at consumption.
    let stale = selected.resource(roles[0]).unwrap().clone();
    let captured_generation = space.memory().mapping_generation().get();
    let base = space.memory().address(address).unwrap();
    space.unmap(base).unwrap();
    space.reserve(1, 4096, 1, address).unwrap();
    space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(2),
            backing: crate::test_memory::ranges(&allocation),
            backing_offset: 0,
            size: 4096,
            allocation_alignment: 4096,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: Some(base),
        })
        .unwrap();
    let key = ResolvedViewKey {
        address_space: space.memory().id(),
        permissions: MemoryPermissions::READ.bits(),
        definition: ResolvedViewDefinition::Sampled {
            bytes: tic
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            dimension: MaxwellThreeDTextureDimension::Two,
        },
    };
    cache
        .retained_backings
        .remember_view(key, stale, captured_generation);
    assert!(cache.retained_backings.view(&key, space.memory()).is_none());
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "opt-in production texture binding CPU/allocation measurement"]
fn texture_binding_throughput() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    let (allocation, space, address) = crate::shader::test_support::mapped_memory();
    for index in 0..2 {
        let image = address + 0x400 + index * 0x100;
        let words = [
            0x74e2_4915u32,
            image as u32,
            (image >> 32) as u32 | (2 << 21),
            1,
            3 | (1 << 23),
            3 | (1 << 31),
            0,
            0,
        ];
        crate::test_memory::write(
            &allocation,
            0x100 + index * 32,
            &words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x2380, 128),
        (0x2384, (address >> 32) as u32),
        (0x2388, address as u32),
        (0x2410, 1),
        (0x1574, (address >> 32) as u32),
        (0x1578, address as u32 + 0x100),
        (0x157c, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let roles = [MaxwellThreeDResourceRole::SampledImage {
        texture: MaxwellThreeDTextureReference::new(0, 0, 0),
        dimension: MaxwellThreeDTextureDimension::Two,
        with_sampler: false,
    }];
    let mut cache = MaxwellThreeDResolvedResourceCache::new(1);
    let mut access = crate::test_memory::access(space.memory().machine());
    let iterations = std::env::var("NIXE_TEXTURE_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);
    for alternating in [false, true] {
        for _ in 0..100 {
            for handle in 0u32..2 {
                crate::test_memory::write(&allocation, 0, &handle.to_le_bytes()).unwrap();
                cache
                    .select(&roles, &[])
                    .resolve(channel.three_d(), space.memory(), 4096, &mut access)
                    .unwrap();
            }
        }
        let cpu_ns = || -> u64 {
            std::fs::read_to_string("/proc/thread-self/schedstat")
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap()
        };
        let cpu_start = cpu_ns();
        let start = std::time::Instant::now();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        for index in 0..iterations {
            if alternating {
                crate::test_memory::write(&allocation, 0, &(index as u32 & 1).to_le_bytes())
                    .unwrap();
            }
            std::hint::black_box(
                cache
                    .select(&roles, &[])
                    .resolve(channel.three_d(), space.memory(), 4096, &mut access)
                    .unwrap(),
            );
        }
        let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        let elapsed_ns = start.elapsed().as_nanos();
        let cpu_ns = cpu_ns() - cpu_start;
        println!(
            "texture_alternating={alternating} iterations={iterations} cpu_ns_per_draw={:.3} elapsed_ns_per_draw={:.3} allocations={allocations}",
            cpu_ns as f64 / iterations as f64,
            elapsed_ns as f64 / iterations as f64
        );
    }
}
