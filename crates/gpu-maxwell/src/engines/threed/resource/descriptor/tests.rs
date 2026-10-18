use super::*;
use nixe_memory::CanonicalAllocation;

fn memory() -> (CanonicalAllocation, CanonicalBackingRange) {
    let allocation = CanonicalAllocation::zeroed(8192, 4096).unwrap();
    let range = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    (allocation, range)
}
#[test]
fn cached_descriptors_keep_old_versions_and_refresh_only_identical_bytes() {
    let (allocation, range) = memory();
    let mut cache = Cache::default();
    let first = cache.read(&range, 64, 32).unwrap();
    let repeated = cache.read(&range, 64, 32).unwrap();
    assert!(Arc::ptr_eq(&first, &repeated));
    allocation.write(256, &[9]).unwrap();
    let disjoint = cache.read(&range, 64, 32).unwrap();
    assert!(Arc::ptr_eq(&first, &disjoint));
    allocation.write(64, &[7]).unwrap();
    let changed = cache.read(&range, 64, 32).unwrap();
    assert_eq!(changed.bytes[0], 7);
    assert!(first.range.shares_layout(&changed.range));
    assert!(!first.remains_current().unwrap());
    assert_eq!(first.bytes, [0; 32]);
    assert!(changed.remains_current().unwrap());
    allocation.write(64, &[0]).unwrap();
    assert!(first.remains_current().unwrap());
    assert!(!changed.remains_current().unwrap());
}
#[test]
fn committed_cross_page_descriptors_keep_independent_byte_versions() {
    let (allocation, range) = memory();
    let mut cache = Cache::default();
    let original = cache.read(&range, 4080, 32).unwrap();
    allocation.write(4094, &[1, 2, 3, 4]).unwrap();
    let first = cache.read(&range, 4080, 32).unwrap();
    assert_eq!(&first.bytes[14..18], &[1, 2, 3, 4]);
    assert!(!original.remains_current().unwrap());
    allocation.write(4095, &[8]).unwrap();
    let second = cache.read(&range, 4080, 32).unwrap();
    assert_eq!(&second.bytes[14..18], &[1, 8, 3, 4]);
    assert_eq!(&first.bytes[14..18], &[1, 2, 3, 4]);
    assert!(!first.remains_current().unwrap());
    assert!(second.remains_current().unwrap());
}
#[test]
fn bounded_eviction_keeps_retained_snapshots_and_distinct_physical_mappings() {
    let (_, a) = memory();
    let (other, b) = memory();
    other.write(0, &[7]).unwrap();
    let mut cache = Cache {
        entries: nixe_gpu::WorkingSet::new(2, 64 * 1024),
    };
    let retained = cache.read(&a, 0, 4).unwrap();
    let mapped = cache.read(&b, 0, 4).unwrap();
    assert_eq!(mapped.bytes[0], 7);
    for offset in (32..256).step_by(32) {
        cache.read(&a, offset, 32).unwrap();
    }
    assert!(cache.entries.usage().0 <= 2);
    assert!(retained.remains_current().unwrap());
    assert_eq!(retained.bytes[0], 0);
    assert_eq!(cache.read(&b, 0, 4).unwrap().bytes[0], 7);
    CanonicalBackingRange::invalidate_visibility_ranges([&b]).unwrap();
    assert!(!mapped.remains_current().unwrap());
    assert!(cache.read(&b, 0, 4).is_err());
}
#[test]
fn unchanged_cached_descriptor_reads_allocate_nothing_with_retained_versions() {
    use crate::engines::tests::register_benchmark::ALLOCATIONS;
    let (_, range) = memory();
    let mut cache = Cache::default();
    let retained = cache.read(&range, 16, 32).unwrap();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    for _ in 0..1000 {
        std::hint::black_box(cache.read(&range, 16, 32).unwrap());
    }
    let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert_eq!(count, 0);
    assert!(retained.remains_current().unwrap());
}
#[test]
fn outstanding_device_producer_invalidates_descriptor_without_early_readback() {
    use nixe_memory::{
        CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
        DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Producer(AtomicUsize);
    impl VisibilityCoordinator for Producer {
        fn cache_cpu_page(
            &self,
            _: DeviceVisibilityRequest,
            _: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }
        fn make_cpu_visible(
            &self,
            request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(vec![7; request.size].into_boxed_slice())
        }
    }
    let (_, range) = memory();
    let mut cache = Cache::default();
    let before = cache.read(&range, 16, 32).unwrap();
    let producer = Arc::new(Producer(AtomicUsize::new(0)));
    let write = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(1),
    )
    .unwrap();
    CanonicalBackingRange::publish_device_writes([(&range, write)], producer.clone()).unwrap();
    assert!(!before.remains_current().unwrap());
    assert_eq!(producer.0.load(Ordering::Relaxed), 0);
    let after = cache.read(&range, 16, 32).unwrap();
    assert_eq!(after.bytes, [7; 32]);
    assert_eq!(producer.0.load(Ordering::Relaxed), 1);
    assert!(!before.remains_current().unwrap());
}
