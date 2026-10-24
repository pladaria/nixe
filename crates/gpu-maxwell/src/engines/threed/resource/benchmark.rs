//! Same descriptor-consumer harness for the frozen and candidate source trees.
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
        .buffer(
            MaxwellThreeDResourceRole::TextureHeaders,
            MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32),
            8192,
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
            .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
            .unwrap()
            .clone();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            if cold {
                cache = MaxwellThreeDResolvedResourceCache::new(1);
            }
            let bindings = cache
                .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
                .unwrap()
                .clone();
            if !cold {
                assert!(first.same_bindings(&bindings));
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
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let repeated = cache
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    let warm = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert!(first.same_bindings(&repeated));
    program_three_d(&mut channel, 0x2410, 17);
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let unused = cache
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    let unused_count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert_eq!(
        first.resource(roles[0]).unwrap().source(),
        unused.resource(roles[0]).unwrap().source()
    );
    cache
        .resolve(channel.three_d(), space.memory(), &[], 4096, &mut access)
        .unwrap();
    program_three_d(&mut channel, 0x2410, 16);
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let after = cache
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    let after_count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert_eq!((warm, unused_count, after_count), (0, 0, 0));
    assert!(first.same_bindings(&unused));
    assert!(unused.same_bindings(&after));
    println!(
        "AUDIT warm={warm}; unused_slot_rebind={unused_count} same_result={}; after_nonconsuming_operation={after_count} same_result={}",
        first.same_bindings(&unused),
        unused.same_bindings(&after)
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
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    program_three_d(&mut channel, 0x2388, address as u32 + 64);
    program_three_d(&mut channel, 0x2410, 17);
    let changed = cache
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    assert!(first.same_resource(&changed, roles[0]));
    assert!(!first.same_resource(&changed, roles[1]));
    assert_eq!(
        changed.resource(roles[1]).unwrap().source().offset().get(),
        address + 64
    );
    // A failed second slot must not leave the first slot's old selection live.
    program_three_d(&mut channel, 0x2388, address as u32 + 128);
    program_three_d(&mut channel, 0x2410, 1);
    program_three_d(&mut channel, 0x2388, address as u32 + 0x2000);
    program_three_d(&mut channel, 0x2410, 17);
    assert!(
        cache
            .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
            .is_err()
    );
    program_three_d(&mut channel, 0x2388, address as u32 + 256);
    program_three_d(&mut channel, 0x2410, 17);
    let changed = cache
        .resolve(channel.three_d(), space.memory(), &roles, 4096, &mut access)
        .unwrap()
        .clone();
    assert_eq!(
        changed.resource(roles[0]).unwrap().source().offset().get(),
        address + 128
    );
    assert_eq!(
        changed.resource(roles[1]).unwrap().source().offset().get(),
        address + 256
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
