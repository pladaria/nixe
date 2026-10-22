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
    let mut builder =
        ResourceBuilder::new(space.memory(), None, Some(&mut cache), &mut memory_access);
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
