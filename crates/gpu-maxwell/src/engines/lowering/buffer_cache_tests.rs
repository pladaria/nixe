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
    let allocation = GpuAllocationDescription::new(bytes.size() as u64, 1).unwrap();
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
fn equal_logical_intervals_preserve_canonical_identity_and_retirement() {
    let first_bytes = CanonicalAllocation::zeroed(4096, 4096).unwrap();
    let second_bytes = CanonicalAllocation::zeroed(4096, 4096).unwrap();
    let mut cache = MaxwellLoweringCache::default();
    // Both views use the same logical allocation interval. Their retained
    // physical pages are different and must never share a backend buffer.
    let first = prepare_slice(&mut cache, &first_bytes, 0);
    let second = prepare_slice(&mut cache, &second_bytes, 0);
    assert_ne!(first, second);
    assert_eq!(prepare_slice(&mut cache, &first_bytes, 0), first);
    assert_eq!(prepare_slice(&mut cache, &second_bytes, 0), second);

    retire_view_dependencies(&[first], &mut cache, &mut Vec::new());
    assert_ne!(prepare_slice(&mut cache, &first_bytes, 0), first);
    assert_eq!(prepare_slice(&mut cache, &second_bytes, 0), second);
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
    assert!(cache.views.values().any(|view| view.dependency == first));
    assert!(cache.views.values().any(|view| view.dependency == read));
    assert!(
        !cache
            .views
            .values()
            .any(|view| view.dependency == overlapping)
    );
    let write = prepare_slice_access(&mut cache, &bytes, 8, true);
    assert_eq!(cache.views.len(), 1);
    assert_eq!(cache.views.first_key_value().unwrap().1.dependency, write);
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
                .values()
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
            .values()
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
            .values()
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
    assert!(cache.views.values().any(|view| view.dependency == next));
    assert!(
        cache
            .views
            .values()
            .any(|view| view.dependency == destination)
    );
}

#[test]
fn configured_budget_retains_constant_buffer_rings_larger_than_256_views() {
    let bytes = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut cache =
        MaxwellLoweringCache::new(GpuCacheConfiguration::new(512, 512, 1, 1, 1).unwrap());
    let destination = prepare_slice_access(&mut cache, &bytes, 0xf000, true);
    let mut ring = Vec::new();
    for index in 0..384 {
        let view = prepare_slice(&mut cache, &bytes, index * 32);
        assert!(
            finish(&mut cache, vec![copy(view, destination)])
                .invalidations
                .is_empty()
        );
        ring.push(view);
    }
    for (index, expected) in ring.into_iter().enumerate() {
        let actual = prepare_slice(&mut cache, &bytes, index as u64 * 32);
        assert_eq!(actual, expected);
        assert!(
            finish(&mut cache, vec![copy(actual, destination)])
                .invalidations
                .is_empty()
        );
    }
}

#[test]
fn inline_ring_writes_batch_only_when_pending_canonical_accesses_are_disjoint() {
    use crate::engines::tests::{program_three_d, three_d_channel};
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };
    use crate::{MaxwellBackendExecution, MaxwellSubmissionExecutionStep};
    for write_offset in [0, 32] {
        let bytes = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        bytes.write(0, &[1; 16]).unwrap();
        let mut cache = MaxwellLoweringCache::default();
        let read = prepare_slice(&mut cache, &bytes, 0);
        let written = prepare_slice(&mut cache, &bytes, 16);
        let later = prepare_slice(&mut cache, &bytes, write_offset);
        let later_written = prepare_slice(&mut cache, &bytes, 48);
        let first_work = finish(&mut cache, vec![copy(read, written)]);
        let later_work = finish(&mut cache, vec![copy(later, later_written)]);
        assert!(!first_work.canonical_accesses().1);
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        // A different logical allocation and VA still name the same physical bytes.
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(99),
                backing: bytes.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
                backing_offset: 0,
                size: 4096,
                allocation_alignment: 4096,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let target = space
            .resolve_range(
                nixe_gpu::GpuVirtualAddress::try_new(mapping.offset().get() + write_offset, 40)
                    .unwrap(),
                16,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x2380, 16);
        let source = channel
            .three_d()
            .shader_bindings()
            .selector()
            .size()
            .source()
            .unwrap();
        let mut execution = MaxwellBackendExecution::begin(FrontendSubmissionId::new(10), None);
        execution.append_steps(vec![
            MaxwellSubmissionExecutionStep::Gpu(first_work),
            MaxwellSubmissionExecutionStep::InlineWrite {
                source,
                target,
                value: vec![2; 16],
            },
        ]);
        // A streaming producer can deliver the following draw in a later prefix.
        let prefix = execution.next_segment(0).unwrap();
        let mut actual = [0; 16];
        bytes.read(write_offset as usize, &mut actual).unwrap();
        if write_offset == 0 {
            let prefix = prefix.expect("an overlapping write must capture its predecessor");
            assert!(!prefix.submission().is_final_segment());
            assert_eq!(
                actual, [1; 16],
                "the earlier read must retain its original version"
            );
            execution.resume_segment();
        } else {
            assert!(
                prefix.is_none(),
                "disjoint streamed slices must not split a batch"
            );
            assert_eq!(actual, [0; 16]);
        }
        execution.append_steps(vec![MaxwellSubmissionExecutionStep::Gpu(later_work)]);
        execution.seal();
        let segment = execution.next_segment(0).unwrap().unwrap();
        assert!(segment.submission().is_final_segment());
        let copies = segment
            .submission()
            .operations()
            .iter()
            .filter(|op| matches!(op.command(), GpuCommand::Copy(_)))
            .count();
        assert_eq!(copies, if write_offset == 0 { 1 } else { 2 });
        bytes.read(write_offset as usize, &mut actual).unwrap();
        assert_eq!(actual, [2; 16]);
        execution.resume_segment();
        assert!(execution.next_segment(0).unwrap().is_none());
    }
}
