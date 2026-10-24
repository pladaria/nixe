use super::*;
use crate::test_memory;
use nixe_memory::{Allocation, Machine};

fn memory() -> (Allocation, Arc<PhysicalMapping>) {
    let machine = test_memory::machine();
    let allocation = test_memory::allocate(&machine, 8192);
    let mapping = test_memory::mapping(&allocation);
    (allocation, mapping)
}
#[test]
fn cached_descriptor_reads_keep_owned_bytes_and_never_resurrect_an_invalid_proof() {
    let (allocation, mapping) = memory();
    let mut access = test_memory::access(allocation.pin().machine());
    let mut cache = Cache::default();
    let first = cache.read(&mapping, 64, 32, &mut access).unwrap();
    assert!(Arc::ptr_eq(
        &first,
        &cache.read(&mapping, 64, 32, &mut access).unwrap()
    ));
    test_memory::write(&allocation, 64, &[7]).unwrap();
    let changed = cache.read(&mapping, 64, 32, &mut access).unwrap();
    assert_eq!(changed.bytes[0], 7);
    assert!(!first.remains_current());
    assert_eq!(first.bytes, [0; 32]);
    assert!(changed.remains_current());
    test_memory::write(&allocation, 64, &[0]).unwrap();
    assert!(!first.remains_current());
    assert!(!changed.remains_current());
    assert_eq!(
        cache.read(&mapping, 64, 32, &mut access).unwrap().bytes,
        [0; 32]
    );
}
#[test]
fn cross_page_descriptors_keep_independent_owned_byte_versions() {
    let (allocation, mapping) = memory();
    let mut access = test_memory::access(allocation.pin().machine());
    let mut cache = Cache::default();
    let original = cache.read(&mapping, 4080, 32, &mut access).unwrap();
    test_memory::write(&allocation, 4094, &[1, 2, 3, 4]).unwrap();
    let first = cache.read(&mapping, 4080, 32, &mut access).unwrap();
    assert_eq!(&first.bytes[14..18], &[1, 2, 3, 4]);
    assert!(!original.remains_current());
    test_memory::write(&allocation, 4095, &[8]).unwrap();
    let second = cache.read(&mapping, 4080, 32, &mut access).unwrap();
    assert_eq!(&second.bytes[14..18], &[1, 8, 3, 4]);
    assert_eq!(&first.bytes[14..18], &[1, 2, 3, 4]);
    assert!(!first.remains_current());
    assert!(second.remains_current());
}
#[test]
fn bounded_descriptor_eviction_keeps_owned_bytes_without_retaining_ram() {
    let (allocation, mapping) = memory();
    let machine = allocation.pin().machine().clone();
    let mut access = test_memory::access(&machine);
    let mut cache = Cache {
        tables: nixe_gpu::WorkingSet::new(2, 128 * 1024),
    };
    let retained = cache.read(&mapping, 0, 4, &mut access).unwrap();
    for offset in (2048..8192).step_by(2048) {
        cache.read(&mapping, offset, 32, &mut access).unwrap();
    }
    assert!(cache.tables.usage().0 <= 2);
    assert_eq!(retained.bytes, [0; 32]);
    drop(allocation);
    assert_eq!(machine.statistics().allocated_bytes, 0);
    assert!(!retained.remains_current());
    assert!(cache.read(&mapping, 0, 4, &mut access).is_err());
}
#[test]
fn unchanged_descriptor_reads_allocate_nothing() {
    use crate::engines::tests::register_benchmark::ALLOCATIONS;
    let (allocation, mapping) = memory();
    let mut access = test_memory::access(allocation.pin().machine());
    let mut cache = Cache::default();
    let retained = cache.read(&mapping, 16, 32, &mut access).unwrap();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    for _ in 0..1000 {
        std::hint::black_box(cache.read(&mapping, 16, 32, &mut access).unwrap());
    }
    assert_eq!(ALLOCATIONS.with(|n| n.replace(None).unwrap()), 0);
    assert!(retained.remains_current());
}
#[test]
fn device_publication_invalidates_descriptors_before_owner_readback() {
    let (allocation, mapping) = memory();
    let machine: Machine = allocation.pin().machine().clone();
    let mut cache = Cache::default();
    let before = cache
        .read(&mapping, 16, 32, &mut test_memory::access(&machine))
        .unwrap();
    machine
        .publish_device_write(&test_memory::ranges(&allocation), || Ok(()))
        .unwrap();
    assert!(!before.remains_current());
    let mut readbacks = 0;
    let mut owner = |ranges: Vec<nixe_memory::AllocationRange>, operation| {
        readbacks += 1;
        machine
            .write_resolved(&ranges, &[7; 32])
            .map_err(Arc::new)?;
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    };
    let after = cache.read(&mapping, 16, 32, &mut owner).unwrap();
    assert_eq!(after.bytes, [7; 32]);
    assert_eq!(readbacks, 1);
    assert!(!before.remains_current());
}
