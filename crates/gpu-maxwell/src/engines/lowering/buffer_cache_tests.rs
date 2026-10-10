//! Ring-buffer slices must not grow the resource cache across rendered frames.
use super::*;
use nixe_gpu::{
    BackingView, BufferDescription, CopyOperation, GpuAllocationDescription, GpuAllocationId,
};
use nixe_memory::{CanonicalAllocation, MemoryPermissions};

fn prepare_slice(
    cache: &mut MaxwellLoweringCache,
    bytes: &CanonicalAllocation,
    offset: u64,
) -> ResourceDependency {
    prepare_slice_access(cache, bytes, offset, false)
}

fn prepare_slice_access(
    cache: &mut MaxwellLoweringCache,
    bytes: &CanonicalAllocation,
    offset: u64,
    writable: bool,
) -> ResourceDependency {
    let allocation = GpuAllocationDescription::new(0x1000, 1).unwrap();
    let canonical = bytes.backing_range(MemoryPermissions::READ_WRITE).unwrap();
    let backing = BackingView::new(
        GpuAllocationId::new(1),
        allocation,
        offset,
        canonical.snapshot_subrange(offset, 16).unwrap(),
    )
    .unwrap();
    buffer::prepare_buffer(
        BufferDescription::new(16).unwrap(),
        allocation,
        backing,
        Arc::from([]),
        writable,
        cache,
        &mut Vec::new(),
        &mut Vec::new(),
        std::iter::empty(),
    )
    .unwrap()
}

#[test]
fn overlapping_read_only_slices_reuse_views_but_writes_reconcile_aliases() {
    let bytes = CanonicalAllocation::zeroed(4096, 4096).unwrap();
    let mut cache = MaxwellLoweringCache::default();
    let first = prepare_slice(&mut cache, &bytes, 0);
    let overlapping = prepare_slice(&mut cache, &bytes, 8);
    assert_eq!(cache.views.len(), 2);
    assert_eq!(prepare_slice(&mut cache, &bytes, 0), first);
    assert_eq!(prepare_slice(&mut cache, &bytes, 8), overlapping);

    finish(&mut cache, vec![copy(first, overlapping)]);
    let read = prepare_slice(&mut cache, &bytes, 16);
    assert!(cache.views.iter().any(|view| view.dependency == first));
    assert!(cache.views.iter().any(|view| view.dependency == read));
    assert!(
        !cache
            .views
            .iter()
            .any(|view| view.dependency == overlapping)
    );
    let write = prepare_slice_access(&mut cache, &bytes, 8, true);
    assert_eq!(cache.views.len(), 1);
    assert_eq!(cache.views[0].dependency, write);
}

fn region(dependency: ResourceDependency) -> BufferRegion {
    let ResourceDependency::Buffer(buffer) = dependency else {
        panic!("buffer slice")
    };
    BufferRegion {
        buffer,
        range: BufferRange::new(0, 16).unwrap(),
    }
}

fn copy(source: ResourceDependency, destination: ResourceDependency) -> GpuOperation {
    GpuOperation::new(
        GpuCommand::Copy(
            CopyOperation::buffer_to_buffer(region(source), region(destination)).unwrap(),
        ),
        [],
        [],
        CapabilityRequirements::none(),
    )
}

fn finish(cache: &mut MaxwellLoweringCache, commands: Vec<GpuOperation>) -> MaxwellLoweredWork {
    finish_lowered_work(
        cache,
        FrontendSubmissionId::new(cache.revision + 1),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        commands,
        Arc::from([]),
    )
    .unwrap()
}

#[test]
fn read_only_buffer_eviction_preserves_gpu_writes_current_descriptors_and_recent_slices() {
    let bytes = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut cache = MaxwellLoweringCache::new(GpuCacheConfiguration::new(6, 2, 1, 1, 1).unwrap());
    let old = prepare_slice(&mut cache, &bytes, 0);
    let recent = prepare_slice(&mut cache, &bytes, 16);
    let written = prepare_slice(&mut cache, &bytes, 32);
    assert!(
        finish(&mut cache, vec![copy(old, written)])
            .invalidations
            .is_empty()
    );
    assert!(
        finish(&mut cache, vec![copy(recent, written)])
            .invalidations
            .is_empty()
    );
    let current = prepare_slice(&mut cache, &bytes, 48);
    let unused = prepare_slice(&mut cache, &bytes, 64);
    let obsolete_table = DescriptorTableId::new(100);
    let current_table = DescriptorTableId::new(101);
    for (id, dependency) in [(obsolete_table, old), (current_table, current)] {
        cache.descriptors.push(DescriptorRecord {
            kinds: Box::new([DescriptorKind::Buffer]),
            bindings: Box::new([0]),
            dependencies: Box::new([dependency]),
            id,
        });
    }
    let operation = GpuOperation::new(
        GpuCommand::Clear(ClearOperation::buffer(region(written), 0).unwrap()),
        [],
        [ResourceDependency::DescriptorTable(current_table)],
        CapabilityRequirements::none(),
    );
    let work = finish(&mut cache, vec![operation]);
    for preserved in [recent, current, written] {
        assert!(
            cache
                .views
                .iter()
                .any(|record| record.dependency == preserved)
        );
        assert!(!work.invalidations.contains(&preserved));
    }
    for retired in [
        unused,
        old,
        ResourceDependency::DescriptorTable(obsolete_table),
    ] {
        assert!(work.invalidations.contains(&retired));
    }
    assert_eq!(cache.views.len(), 3);
    assert_eq!(cache.descriptors.len(), 1);
    assert_eq!(cache.descriptors[0].id, current_table);
    assert!(
        cache
            .accesses
            .iter()
            .all(|(target, _)| target.dependency() != old)
    );
    assert!(
        cache
            .views
            .iter()
            .find(|record| record.dependency == written)
            .unwrap()
            .write_revision
            != 0
    );
    // Eviction drops only derived host views, not the canonical allocation.
    assert_eq!(
        bytes
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap()
            .size(),
        0x1000
    );
    assert_ne!(prepare_slice(&mut cache, &bytes, 0), old);
}

#[test]
fn buffer_slice_cache_remains_bounded_over_repeated_deliveries() {
    let bytes = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut cache = MaxwellLoweringCache::new(GpuCacheConfiguration::new(6, 2, 1, 1, 1).unwrap());
    let written = prepare_slice(&mut cache, &bytes, 0);
    for frame in 1..128 {
        let source = prepare_slice(&mut cache, &bytes, frame * 16);
        let work = finish(&mut cache, vec![copy(source, written)]);
        assert!(cache.views.len() <= 3);
        assert!(!work.invalidations.contains(&source));
        assert!(!work.invalidations.contains(&written));
    }
}

#[test]
fn read_only_ring_eviction_leaves_room_without_exceeding_the_cache_budget() {
    let bytes = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut cache = MaxwellLoweringCache::new(GpuCacheConfiguration::new(32, 16, 1, 1, 1).unwrap());
    let mut current = prepare_slice(&mut cache, &bytes, 0);
    let destination = prepare_slice_access(&mut cache, &bytes, 0xf00, true);
    for offset in (16..=256).step_by(16) {
        current = prepare_slice(&mut cache, &bytes, offset);
    }
    let work = finish(&mut cache, vec![copy(current, destination)]);
    assert_eq!(work.invalidations.len(), 2);
    assert_eq!(
        cache
            .views
            .iter()
            .filter(|view| view.write_revision == 0)
            .count(),
        15
    );
    let next = prepare_slice(&mut cache, &bytes, 272);
    assert!(
        finish(&mut cache, vec![copy(next, destination)])
            .invalidations
            .is_empty()
    );
    let next = prepare_slice(&mut cache, &bytes, 288);
    let work = finish(&mut cache, vec![copy(next, destination)]);
    assert_eq!(work.invalidations.len(), 2);
    assert!(cache.views.iter().any(|view| view.dependency == next));
    assert!(
        cache
            .views
            .iter()
            .any(|view| view.dependency == destination)
    );
}
