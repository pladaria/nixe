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
        cache,
        &mut Vec::new(),
        &mut Vec::new(),
        std::iter::empty(),
    )
    .unwrap()
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
fn work_retains_backing_identity_when_a_new_version_is_retired_in_the_same_work() {
    let bytes = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut cache = MaxwellLoweringCache::default();
    let dependency = prepare_slice(&mut cache, &bytes, 64);
    let view = &cache.views[0];
    let ViewKey::Buffer {
        description,
        buffer_offset,
        backing,
        ..
    } = &view.key
    else {
        panic!("buffer");
    };
    let ResourceDependency::Buffer(id) = dependency else {
        panic!("buffer");
    };
    let creation = BackendResourceCreateInfo::Buffer {
        id,
        description: *description,
        view: Some(BufferView::new(id, *description, *buffer_offset, backing.clone()).unwrap()),
    };
    let work = finish_lowered_work(
        &mut cache,
        FrontendSubmissionId::new(1),
        vec![],
        vec![creation],
        vec![dependency],
        vec![GpuOperation::new(
            GpuCommand::UploadBuffer {
                destination: region(dependency),
                bytes: Arc::from([7; 16]),
            },
            [],
            [],
            CapabilityRequirements::none(),
        )],
        Arc::from([]),
    )
    .unwrap();
    assert_eq!(work.resident_resources.len(), 1);
    assert_eq!(work.resident_resources[0].dependency, dependency);
    assert_eq!(
        work.resident_resources[0].backings[0].allocation_offset(),
        64
    );
    assert_eq!(work.resource_invalidations(), &[dependency]);
    assert_eq!(work.resource_creations()[0].dependency(), dependency);
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
