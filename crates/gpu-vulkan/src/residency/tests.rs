use super::*;
use crate::{ApiPreference, QueueConfiguration, queue::QueueState};
use nixe_memory::{CanonicalAllocation, MemoryPermissions, VisibilityCoordinator};
use std::time::Duration;

#[test]
fn cpu_streaming_buffer_imports_keep_independent_aliases_and_gpu_handoffs() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(4096, 4096, 0x19);
    let a = buffer(&mut cache, backing.clone());
    let b = cache
        .buffer_for(backing.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    for value in 0..16 {
        allocation.write(17, &[value; 37]).unwrap();
        assert_eq!(cache.read_buffer(a, 17, 37).unwrap(), [value; 37]);
        assert_eq!(cache.read_buffer(b, 17, 37).unwrap(), [value; 37]);
    }
    write(&mut cache, a, 17, &[0xa7; 37]);
    assert_eq!(cache.read_buffer(b, 17, 37).unwrap(), [0xa7; 37]);
    assert_eq!(&cpu_read(&mut cache, &backing)[17..54], &[0xa7; 37]);
    allocation.write(20, &[0x42; 7]).unwrap();
    for id in [a, b] {
        let mut expected = [0xa7; 37];
        expected[3..10].fill(0x42);
        assert_eq!(cache.read_buffer(id, 17, 37).unwrap(), expected);
    }
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn deferred_buffer_import_observes_writes_after_preparation() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(4096, 4096, 0x19);
    let prepared = cache
        .buffer_for_deferred_import(backing.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 0);
    allocation.write(0, &[0x37; 4096]).unwrap();
    assert_eq!(cache.read_buffer(prepared, 0, 4096).unwrap(), [0x37; 4096]);
    let producer = buffer(&mut cache, backing.clone());
    write(&mut cache, producer, 0, &[0x51; 4096]);
    let before = cache.statistics().cpu_uploaded_bytes;
    assert_eq!(
        cache
            .buffer_for_deferred_import(backing, vk::BufferUsageFlags::VERTEX_BUFFER)
            .unwrap(),
        prepared
    );
    assert_eq!(cache.statistics().cpu_uploaded_bytes, before);
    assert_eq!(cache.read_buffer(prepared, 0, 4096).unwrap(), [0x51; 4096]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn buffer_mapping_index_preserves_subranges_usage_and_eviction_generations() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(4096, 4096, 0x19);
    let mut ids = Vec::new();
    for offset in (0..2048).step_by(16) {
        ids.push(buffer(&mut cache, whole.logical_view(offset, 16).unwrap()));
    }
    let readonly = allocation.backing_range(MemoryPermissions::READ).unwrap();
    for (index, &id) in ids.iter().enumerate() {
        assert_eq!(
            buffer(
                &mut cache,
                readonly.logical_view(index as u64 * 16, 16).unwrap()
            ),
            id
        );
    }
    let range = whole.logical_view(16, 16).unwrap();
    let vertex = cache
        .buffer_for(range.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    assert_ne!(vertex, ids[1]);
    assert_eq!(buffer(&mut cache, range.clone()), ids[1]);
    allocation.write(16, &[0x37; 16]).unwrap();
    assert_eq!(cache.read_buffer(ids[1], 0, 16).unwrap(), [0x37; 16]);
    assert_eq!(cache.read_buffer(vertex, 0, 16).unwrap(), [0x37; 16]);
    cache.flush().unwrap();
    while cache.evict_one().unwrap() {}
    assert!(cache.buffer(ids[1]).is_err());
    let next = buffer(&mut cache, range);
    assert_ne!(next, ids[1]);
    assert_eq!(cache.read_buffer(next, 0, 16).unwrap(), [0x37; 16]);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn invalidation_union_preserves_holes_and_is_idempotent() {
    let (_, backing) = backing_allocation(128, 128, 0);
    let mut contents = Contents {
        cpu: CanonicalCpuWriteDependency::capture(&backing).unwrap(),
        backing,
        invalid: Vec::new(),
        last_use: None,
        touched: 0,
    };
    let mut expected = [false; 128];
    for added in [
        vec![(16, 8), (40, 16), (100, 4)],
        vec![(8, 12), (24, 16), (55, 10), (104, 8)],
        vec![(12, 4), (44, 4), (106, 2)],
        vec![(0, 128)],
        (0..32).map(|i| (i * 4, 1)).collect(),
        vec![],
    ] {
        for &(start, size) in &added {
            expected[start as usize..(start + size) as usize].fill(true);
        }
        contents.invalidate_intervals(&added);
        contents.invalidate_intervals(&added);
        let mut actual = [false; 128];
        for &(start, size) in &contents.invalid {
            actual[start as usize..(start + size) as usize].fill(true);
        }
        assert_eq!(actual, expected);
        assert!(
            contents
                .invalid
                .windows(2)
                .all(|w| w[0].0 + w[0].1 < w[1].0)
        );
    }
}

#[test]
fn batched_depth_byte_validation_preserves_only_unwritten_intersections() {
    let (_, backing) = backing_allocation(4096, 4096, 0);
    let mut contents = Contents {
        cpu: CanonicalCpuWriteDependency::capture(&backing).unwrap(),
        backing,
        invalid: vec![(0, 4096)],
        last_use: None,
        touched: 0,
    };
    let written: Vec<_> = (0..1024).map(|pixel| (pixel * 4, 3)).collect();
    contents.validate_intervals(&written);
    let padding: Vec<_> = (0..1024).map(|pixel| (pixel * 4 + 3, 1)).collect();
    assert_eq!(contents.invalid, padding);
    contents.validate_intervals(&written);
    assert_eq!(contents.invalid, padding);
    contents.validate_intervals(&[(2, 31), (100, 49), (4080, 16)]);
    let mut actual = vec![false; 4096];
    for &(start, length) in &contents.invalid {
        actual[start as usize..(start + length) as usize].fill(true);
    }
    for (offset, invalid) in actual.into_iter().enumerate() {
        assert_eq!(
            invalid,
            offset % 4 == 3
                && !(2..33).contains(&offset)
                && !(100..149).contains(&offset)
                && !(4080..4096).contains(&offset)
        );
    }
}
mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}
mod gate {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/gate.rs"
    ));
}
pub(crate) fn cache(api: ApiPreference, budget: u64) -> Option<VulkanResidency> {
    let mut device = support::device_for_test(api)?;
    device.submission = QueueState::new(
        Arc::clone(&device.owner),
        device.queues().graphics_compute_family,
        QueueConfiguration {
            command_slots: 4,
            arena_bytes: 16 * 1024,
            overflow_bytes: 64 * 1024,
        },
    )
    .unwrap();
    Some(VulkanResidency::new(device, budget).unwrap())
}
pub(crate) fn backing_allocation(
    size: usize,
    page_size: usize,
    byte: u8,
) -> (CanonicalAllocation, CanonicalBackingRange) {
    let allocation = CanonicalAllocation::zeroed(size, page_size).unwrap();
    allocation.write(0, &vec![byte; size]).unwrap();
    let range = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    (allocation, range)
}
pub(crate) fn cpu_read(cache: &mut VulkanResidency, range: &CanonicalBackingRange) -> Vec<u8> {
    let mut data = vec![0; range.size() as usize];
    range
        .read_with_resolver(0, &mut data, &mut |_, request| {
            cache
                .materialize_cpu_page(request)
                .map_err(|error| nixe_memory::VisibilityCoordinatorError::new(error.to_string()))
        })
        .unwrap();
    data
}
pub(super) fn buffer(cache: &mut VulkanResidency, range: CanonicalBackingRange) -> BufferId {
    cache
        .buffer_for(range, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
}
fn write(cache: &mut VulkanResidency, id: BufferId, offset: u64, bytes: &[u8]) -> WorkTicket {
    let progress = cache.write_buffer(id, offset, bytes).unwrap();
    assert_eq!(progress.consumed, bytes.len());
    progress.work
}
fn exercise_buffer_coherence(api: ApiPreference) {
    let Some(mut cache) = cache(api, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(3 * 4096, 4096, 0x19);
    let a = buffer(&mut cache, whole.clone());
    assert_eq!(cache.statistics().cpu_uploaded_bytes, whole.size());
    assert_eq!(
        cache.device.queue_usage().arena_capacity,
        16 * 1024,
        "uploads alone create no readback arena"
    );
    let old = cache.statistics();
    for _ in 0..100 {
        assert_eq!(buffer(&mut cache, whole.clone()), a);
    }
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        old.cpu_uploaded_bytes
    );
    allocation.write(4096 + 100, &[0x63; 23]).unwrap();
    cache.ensure_buffer(a).unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes - old.cpu_uploaded_bytes,
        4096
    );
    write(&mut cache, a, 101, &[0xaa; 7]);
    cache.flush().unwrap();
    let b = buffer(&mut cache, whole.snapshot_subrange(80, 80).unwrap());
    assert_eq!(
        cache.statistics().canonical_readback_bytes,
        0,
        "new alias must use GPU producers"
    );
    let data = cache.read_buffer(b, 0, 80).unwrap();
    let mut expected = vec![0x19; 80];
    expected[21..28].fill(0xaa);
    assert_eq!(data, expected);
    // Independent producer on the same conservative page.
    write(&mut cache, b, 50, &[0xbb; 9]);
    cache.flush().unwrap();
    let before = cache.statistics().native_readback_bytes;
    let data = cpu_read(&mut cache, &whole.snapshot_subrange(0, 4096).unwrap());
    assert_eq!(&data[101..108], &[0xaa; 7]);
    assert_eq!(&data[130..139], &[0xbb; 9]);
    assert!(data[..101].iter().all(|byte| *byte == 0x19));
    assert_eq!(
        cache.statistics().native_readback_bytes - before,
        16,
        "CPU demand copies exact authoritative intervals"
    );
    assert_eq!(cache.statistics().canonical_readback_bytes, 16);
    allocation.write(132, &[0xdd; 2]).unwrap();
    cache.ensure_buffer(a).unwrap();
    assert_eq!(
        &cache.read_buffer(a, 130, 9).unwrap(),
        &[0xbb, 0xbb, 0xdd, 0xdd, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb]
    );
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn buffer_coherence_core13() {
    exercise_buffer_coherence(ApiPreference::Auto);
}
#[test]
fn buffer_coherence_khr12() {
    exercise_buffer_coherence(ApiPreference::Compatibility12);
}

#[test]
fn cpu_snapshots_exclude_gpu_holes_and_preserve_repeated_dirty_pages() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (a, range_a) = backing_allocation(4096, 4096, 0x11);
    let (_, range_b) = backing_allocation(4096, 4096, 0x22);
    let (_, range_c) = backing_allocation(4096, 4096, 0x33);
    let gpu = buffer(&mut cache, range_b.snapshot_subrange(0, 32).unwrap());
    write(&mut cache, gpu, 0, &[0x99; 32]);
    cache.flush().unwrap();
    let parts = [&range_a, &range_b, &range_c, &range_a]
        .map(|range| range.snapshot_subrange(0, 32).unwrap());
    let mixed = CanonicalBackingRange::new(
        parts
            .iter()
            .flat_map(|range| range.segments().iter().cloned())
            .collect(),
    )
    .unwrap();
    let dependency = CanonicalCpuWriteDependency::capture(&mixed).unwrap();
    assert_eq!(
        cache
            .cpu_snapshots(&mixed, &dependency, &[(0, 128)])
            .unwrap(),
        vec![
            (0, vec![0x11; 32].into_boxed_slice()),
            (
                64,
                [vec![0x33; 32], vec![0x11; 32]].concat().into_boxed_slice()
            ),
        ]
    );
    a.write(7, &[0x77; 4]).unwrap();
    let mut updated = vec![0x11; 32];
    updated[7..11].fill(0x77);
    // Forced windows need not arrive sorted and can cross GPU-owned holes.
    assert_eq!(
        cache
            .cpu_snapshots(&mixed, &dependency, &[(110, 10), (4, 70)])
            .unwrap(),
        vec![
            (0, updated.clone().into_boxed_slice()),
            (64, vec![0x33; 10].into_boxed_slice()),
            (96, updated.into_boxed_slice()),
        ]
    );
    assert!(
        cache
            .cpu_snapshots(&mixed, &dependency, &[])
            .unwrap()
            .is_empty()
    );
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    assert_eq!(cache.statistics().native_readback_bytes, 0);
    assert!(matches!(
        range_b.pages().next().unwrap().visibility_state(),
        VisibilityState::GpuNewer { .. }
    ));
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn cpu_snapshots_merge_adjacent_forced_windows_and_rearm_full_snapshots() {
    let Some(cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(4096, 4096, 0);
    let mut expected = (0..1024).map(|index| index as u8).collect::<Vec<_>>();
    allocation.write(0, &expected).unwrap();
    let range = whole.snapshot_subrange(0, 1024).unwrap();
    let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
    assert_eq!(
        cache
            .cpu_snapshots(&range, &dependency, &[(0, 1024)])
            .unwrap(),
        vec![(0, expected.clone().into_boxed_slice())]
    );
    assert!(
        cache
            .cpu_snapshots(&range, &dependency, &[])
            .unwrap()
            .is_empty()
    );
    let fragments = (0..128)
        .rev()
        .map(|index| (index * 8, 8))
        .collect::<Vec<_>>();
    assert_eq!(
        cache
            .cpu_snapshots(&range, &dependency, &fragments)
            .unwrap(),
        vec![(0, expected.clone().into_boxed_slice())]
    );
    allocation.write(99, &[0xef; 5]).unwrap();
    expected[99..104].fill(0xef);
    assert_eq!(
        cache.cpu_snapshots(&range, &dependency, &[]).unwrap(),
        vec![(0, expected.into_boxed_slice())]
    );
    assert!(
        cache
            .cpu_snapshots(&range, &dependency, &[])
            .unwrap()
            .is_empty()
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn fragmented_repeated_canonical_offsets_and_same_buffer_copies() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (a, range_a) = backing_allocation(4096, 4096, 0x31);
    let (_, range_b) = backing_allocation(4096, 4096, 0x74);
    let segments = [
        range_a.snapshot_subrange(100, 32).unwrap(),
        range_b.snapshot_subrange(200, 32).unwrap(),
        range_a.snapshot_subrange(100, 32).unwrap(),
    ];
    let range = CanonicalBackingRange::new(
        segments
            .iter()
            .flat_map(|range| range.segments().iter().cloned())
            .collect(),
    )
    .unwrap();
    let id = buffer(&mut cache, range.clone());
    a.write(105, &[0x55; 4]).unwrap();
    cache.ensure_buffer(id).unwrap();
    let bytes = cache.read_buffer(id, 0, 96).unwrap();
    assert_eq!(&bytes[5..9], &[0x55; 4]);
    assert_eq!(&bytes[69..73], &[0x55; 4]);
    write(&mut cache, id, 0, &[0x11; 32]);
    write(&mut cache, id, 64, &[0x22; 32]);
    cache.flush().unwrap();
    let bytes = cache.read_buffer(id, 0, 96).unwrap();
    assert_eq!(&bytes[..32], &[0x22; 32]);
    assert_eq!(&bytes[64..], &[0x22; 32]);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    assert_eq!(
        cpu_read(&mut cache, &range_a.snapshot_subrange(100, 32).unwrap()),
        vec![0x22; 32]
    );
    cache.copy_buffer(id, 0, id, 1, 12).unwrap();
    cache.flush().unwrap();
    assert_eq!(cache.read_buffer(id, 1, 12).unwrap(), vec![0x22; 12]);
    drop(cache);
    validation.check().unwrap();
}
fn rgba_image() -> (NativeImageDescription, Vec<ImagePlane>) {
    (
        NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: 4,
                height: 3,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
        },
        vec![ImagePlane {
            offset: 32,
            row_pitch: 24,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent: vk::Extent3D {
                width: 4,
                height: 3,
                depth: 1,
            },
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        }],
    )
}
#[test]
fn image_publication_refreshes_late_aliases_after_foreign_writes_and_cpu_visibility() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(backing.clone(), description, planes.clone())
        .unwrap();
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [1., 0., 0., 1.],
            },
        )
        .unwrap();
    // Populate coverage before these bindings exist: the next publication must
    // rebuild alias invalidation even though the physical footprint is unchanged.
    let alias = buffer(&mut cache, backing.snapshot_subrange(0, 128).unwrap());
    let mut other_description = description;
    other_description.format = vk::Format::B8G8R8A8_UNORM;
    let other = cache
        .image_for(backing.clone(), other_description, planes)
        .unwrap();
    for color in [[0., 1., 0., 1.], [0., 0., 1., 1.], [1., 1., 1., 1.]] {
        let version = cache.image(image).unwrap().version;
        cache
            .clear_color_image(image, 0, vk::ClearColorValue { float32: color })
            .unwrap();
        assert_eq!(cache.image(image).unwrap().version, version + 1);
        assert!(!cache.buffer(alias).unwrap().contents.invalid.is_empty());
        assert!(!cache.image(other).unwrap().contents.invalid.is_empty());
        let pixel = color.map(|channel| (channel * 255.) as u8);
        let mut expected = vec![0x41; 128];
        for row in 0..3 {
            for x in 0..4 {
                let start = 32 + row * 24 + x * 4;
                expected[start..start + 4].copy_from_slice(&pixel);
            }
        }
        assert_eq!(cache.read_buffer(alias, 0, 128).unwrap(), expected);
        // Expand authority with a foreign buffer writer, then acknowledge a
        // real CPU read. A later image write must republish its latest bytes.
        write(&mut cache, alias, 36, &[0xaa; 4]);
        expected[36..40].fill(0xaa);
        assert_eq!(
            cpu_read(&mut cache, &backing.snapshot_subrange(0, 128).unwrap()),
            expected
        );
        cache.drain_notifications();
    }
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn image_buffer_aliases_compatible_views_and_partial_readback() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(backing.clone(), description, planes.clone())
        .unwrap();
    let view = ImageViewDescription {
        format: vk::Format::R8G8B8A8_SRGB,
        view_type: vk::ImageViewType::TYPE_2D,
        aspects: vk::ImageAspectFlags::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
    };
    let v = cache.image_view(image, view).unwrap();
    assert_eq!(cache.image_view(image, view).unwrap(), v);
    assert_eq!(
        cache.images[image.slot]
            .entry
            .as_ref()
            .unwrap()
            .native_views
            .len(),
        1
    );
    let (_, source_backing) = backing_allocation(4096, 4096, 0x87);
    let source = buffer(&mut cache, source_backing);
    cache.write_image_from_buffer(image, source, 0).unwrap();
    cache.flush().unwrap();
    let alias = buffer(&mut cache, backing.snapshot_subrange(30, 70).unwrap());
    let bytes = cache.read_buffer(alias, 0, 70).unwrap();
    let mut expected = vec![0x41; 70];
    for row in 0..3 {
        expected[2 + row * 24..18 + row * 24].fill(0x87);
    }
    assert_eq!(bytes, expected);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    write(&mut cache, alias, 10, &[0xa3; 5]);
    cache.flush().unwrap();
    cache.ensure_image(image).unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[40..45], &[0xa3; 5]);
    for row in 0..3 {
        assert!(
            bytes[48 + row * 24..56 + row * 24]
                .iter()
                .all(|byte| *byte == 0x41)
        );
    }
    assert_eq!(cache.statistics().canonical_readback_bytes, 48);
    allocation.write(33, &[0xf1; 3]).unwrap();
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cache.image_for(backing, description, planes).unwrap(),
        image
    );
    assert_eq!(
        cache.image_view(image, view).unwrap(),
        v,
        "content changes preserve views"
    );
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn budget_retirement_generation_reuse_and_authority_pinning() {
    let Some(mut cache) = cache(ApiPreference::Auto, 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, range) = backing_allocation(4096, 4096, 0x29);
    let old = buffer(&mut cache, range.clone());
    let charged = cache.allocated_bytes();
    assert!(charged >= 4096);
    // A real external semaphore holds accepted work in flight deterministically.
    let mut gate = gate::Gate::new(&cache.device);
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let work = unsafe {
        cache
            .device
            .submission
            .flush(cache.device.queue, &cache.device.dispatch, &waits, &[])
    }
    .unwrap()
    .unwrap();
    gate.last_use = work.tick();
    assert!(cache.evict_one().unwrap());
    assert_eq!(
        cache.allocated_bytes(),
        charged,
        "retired in-flight allocation remains charged"
    );
    assert!(cache.ensure_buffer(old).is_err());
    gate.signal();
    cache
        .device
        .wait_for_work(work, Duration::from_secs(5))
        .unwrap();
    assert_eq!(cache.allocated_bytes(), 0);
    let new = buffer(&mut cache, range.clone());
    assert_eq!(new.slot, old.slot);
    assert_ne!(new.generation, old.generation);
    write(&mut cache, new, 11, &[0x88; 5]);
    cache.flush().unwrap();
    assert!(
        !cache.evict_one().unwrap(),
        "device-only authoritative bytes cannot be discarded"
    );
    let allocated = cache.allocated_bytes();
    cache.budget.limit = allocated;
    let (_, other) = backing_allocation(4096, 4096, 0x45);
    assert!(matches!(
        cache.buffer_for(other.clone(), vk::BufferUsageFlags::STORAGE_BUFFER),
        Err(VulkanError::ResidencyPressure { .. })
    ));
    assert_eq!(
        cache.statistics().canonical_readback_bytes,
        0,
        "allocation pressure must not read the whole memory"
    );
    assert_eq!(&cpu_read(&mut cache, &range)[11..16], &[0x88; 5]);
    assert!(cache.evict_one().unwrap());
    cache.device.poll_progress().unwrap();
    for _ in 0..64 {
        let id = buffer(&mut cache, other.clone());
        assert!(cache.ensure_buffer(old).is_err());
        let work = cache.flush().unwrap().unwrap();
        cache
            .device
            .wait_for_work(work, Duration::from_secs(5))
            .unwrap();
        assert_eq!(cache.read_buffer(id, 0, 13).unwrap(), vec![0x45; 13]);
        assert!(cache.evict_one().unwrap());
        cache.device.poll_progress().unwrap();
        assert_eq!(cache.allocated_bytes(), 0);
    }
    drop(gate);
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn presentation_lease_pins_version_until_last_reference() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(4096, 4096, 0x29);
    let (description, planes) = rgba_image();
    let image = cache.image_for(backing, description, planes).unwrap();
    let lease = cache.lease_image(image).unwrap();
    let clone = lease.clone();
    while cache.evict_one().unwrap() {}
    assert!(cache.image(image).is_ok());
    allocation.write(35, &[0x55; 4]).unwrap();
    assert!(
        cache.ensure_image(image).is_err(),
        "leased bytes cannot be overwritten"
    );
    drop(lease);
    cache.drain_notifications();
    assert!(cache.ensure_image(image).is_err());
    drop(clone);
    cache.drain_notifications();
    cache.ensure_image(image).unwrap();
    assert!(cache.image(image).unwrap().version > 1);
    while cache.evict_one().unwrap() {}
    assert!(cache.image(image).is_err());
    drop(cache);
    validation.check().unwrap();
}
#[test]
fn stale_cpu_acknowledgement_cannot_release_newer_producer() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, range) = backing_allocation(4096, 4096, 0x31);
    let id = buffer(&mut cache, range.clone());
    let first = write(&mut cache, id, 11, &[0x15; 4]);
    cache.flush().unwrap();
    let old = CpuVisibilityRequest {
        page: range.segments()[0].page(),
        size: 4096,
        device: NonCpuDeviceId::new(cache.instance.get()),
        visible_at: DeviceVisibilityPoint::new(first.tick()),
    };
    cache.materialize_cpu_page(old).unwrap();
    let second = write(&mut cache, id, 12, &[0x25; 4]);
    cache.flush().unwrap();
    let newer = CpuVisibilityRequest {
        visible_at: DeviceVisibilityPoint::new(second.tick()),
        ..old
    };
    cache.materialize_cpu_page(newer).unwrap();
    cache.visibility.release_cpu_page(old).unwrap();
    cache.drain_notifications();
    assert!(cache.authoritative(Resource::Buffer(id)));
    let bytes = cpu_read(&mut cache, &range);
    assert_eq!(&bytes[11..16], &[0x15, 0x25, 0x25, 0x25, 0x25]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn staging_pressure_resumes_partial_cpu_uploads_and_gpu_prefixes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    cache.device.submission = QueueState::new(
        Arc::clone(&cache.device.owner),
        cache.device.queues().graphics_compute_family,
        QueueConfiguration {
            command_slots: 2,
            arena_bytes: 512,
            overflow_bytes: 0,
        },
    )
    .unwrap();
    let (allocation, range) = backing_allocation(4096, 4096, 0x34);
    let mut pressures = 0;
    let id = loop {
        match cache.buffer_for(range.clone(), vk::BufferUsageFlags::STORAGE_BUFFER) {
            Ok(id) => break id,
            Err(VulkanError::QueuePressure { .. }) => {
                pressures += 1;
                let work = cache.flush().unwrap().unwrap();
                cache
                    .device
                    .wait_for_work(work, Duration::from_secs(5))
                    .unwrap();
            }
            Err(error) => panic!("unexpected upload error: {error}"),
        }
        assert!(pressures <= 8, "a resumed upload must make progress");
    };
    assert_eq!(pressures, 7);
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 4096);
    assert_eq!(cache.read_buffer(id, 0, 4096).unwrap(), vec![0x34; 4096]);
    allocation.write(100, &[0x56; 3]).unwrap();
    while let Err(error) = cache.ensure_buffer(id) {
        assert!(matches!(error, VulkanError::QueuePressure { .. }));
        let work = cache.flush().unwrap().unwrap();
        cache
            .device
            .wait_for_work(work, Duration::from_secs(5))
            .unwrap();
    }
    assert_eq!(cache.read_buffer(id, 100, 3).unwrap(), vec![0x56; 3]);
    let bytes = vec![0x99; 2048];
    let mut consumed = 0;
    while consumed < bytes.len() {
        let progress = cache
            .write_buffer(id, consumed as u64, &bytes[consumed..])
            .unwrap();
        assert_eq!(progress.consumed, 512);
        consumed += progress.consumed;
        assert_eq!(cache.flush().unwrap(), Some(progress.work));
        cache
            .device
            .wait_for_work(progress.work, Duration::from_secs(5))
            .unwrap();
    }
    let before = cache.statistics().canonical_readback_bytes;
    cache.preserve_buffer(id).unwrap();
    assert_eq!(cache.statistics().canonical_readback_bytes - before, 2048);
    assert_eq!(&cpu_read(&mut cache, &range)[..2048], &bytes);
    assert!(cache.evict_one().unwrap());
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn native_compressed_blocks_mips_and_layers_preserve_exact_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x19);
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::BC1_RGBA_UNORM_BLOCK,
        extent: vk::Extent3D {
            width: 7,
            height: 5,
            depth: 1,
        },
        mip_levels: 2,
        array_layers: 2,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        flags: vk::ImageCreateFlags::empty(),
    };
    let mut planes = Vec::new();
    let mut offset = 0;
    for layer in 0..2 {
        for mip in 0..2 {
            let extent = vk::Extent3D {
                width: (7 >> mip).max(1),
                height: (5 >> mip).max(1),
                depth: 1,
            };
            planes.push(ImagePlane {
                offset,
                row_pitch: 16,
                mip_level: mip,
                array_layer: layer,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 4,
                    height: 4,
                },
                bytes_per_block: 8,
            });
            offset += 64;
        }
    }
    let image = cache
        .image_for(backing.clone(), description, planes.clone())
        .unwrap();
    let (_, source) = backing_allocation(4096, 4096, 0x71);
    let source = buffer(&mut cache, source);
    cache.write_image_from_buffer(image, source, 0).unwrap();
    cache.flush().unwrap();
    // Splitting one BC block requests exactly three bytes, realigned on GPU.
    let alias = buffer(&mut cache, backing.snapshot_subrange(3, 3).unwrap());
    assert_eq!(cache.read_buffer(alias, 0, 3).unwrap(), vec![0x71; 3]);
    let before = cache.device.queue_usage().submitted_tick;
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[3..6], &[0x71; 3]);
    for plane in planes {
        for row in 0..plane.rows() {
            let start = (plane.offset + plane.row_pitch * u64::from(row)) as usize;
            assert!(
                bytes[start..start + plane.row_bytes() as usize]
                    .iter()
                    .all(|byte| *byte == 0x71)
            );
        }
    }
    assert_eq!(
        cache.device.queue_usage().submitted_tick - before,
        1,
        "all page producer rows share one demanded readback"
    );
    assert_eq!(cache.statistics().canonical_readback_bytes, 80);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn cpu_visibility_routes_to_the_single_owner() {
    use nixe_gpu::BackendVisibilityRequester;
    use nixe_memory::VisibilityCoordinatorError;
    type Reply = mpsc::SyncSender<Result<Box<[u8]>, VisibilityCoordinatorError>>;
    struct Requester(mpsc::Sender<(CpuVisibilityRequest, Reply)>);
    impl BackendVisibilityRequester for Requester {
        fn make_cpu_visible(
            &self,
            request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            let (send, receive) = mpsc::sync_channel(1);
            self.0.send((request, send)).unwrap();
            receive.recv().unwrap()
        }
    }
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x25);
    let id = buffer(&mut cache, backing.clone());
    write(&mut cache, id, 9, &[0x43; 4]);
    cache.flush().unwrap();
    let (send, receive) = mpsc::channel();
    cache
        .visibility
        .bind_requester(Arc::new(Requester(send)))
        .unwrap();
    let worker = std::thread::spawn(move || {
        let mut bytes = [0; 4];
        backing.read(9, &mut bytes).unwrap();
        bytes
    });
    let (request, reply) = receive.recv_timeout(Duration::from_secs(5)).unwrap();
    reply
        .send(
            cache
                .materialize_cpu_page(request)
                .map_err(|error| VisibilityCoordinatorError::new(error.to_string())),
        )
        .unwrap();
    assert_eq!(worker.join().unwrap(), [0x43; 4]);
    cache.drain_notifications();
    assert!(!cache.authoritative(Resource::Buffer(id)));
    assert!(cache.visibility.pages.lock().unwrap().is_empty());
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn repeated_image_rows_and_gpu_padding_writes_keep_texel_identity() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, original) = backing_allocation(4096, 4096, 0x31);
    let part = original.snapshot_subrange(100, 16).unwrap();
    let repeated = CanonicalBackingRange::new(
        part.segments()
            .iter()
            .chain(part.segments())
            .cloned()
            .collect(),
    )
    .unwrap();
    let (mut description, mut planes) = rgba_image();
    description.extent.height = 2;
    planes[0].offset = 0;
    planes[0].row_pitch = 16;
    planes[0].extent.height = 2;
    let image = cache.image_for(repeated, description, planes).unwrap();
    let (source_allocation, source_backing) = backing_allocation(4096, 4096, 0x11);
    source_allocation.write(16, &[0x22; 16]).unwrap();
    let source = buffer(&mut cache, source_backing);
    cache.write_image_from_buffer(image, source, 0).unwrap();
    cache.flush().unwrap();
    assert!(
        !cache.image(image).unwrap().contents.invalid.is_empty(),
        "the earlier physical alias must observe the later row producer"
    );
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cpu_read(&mut cache, &original.snapshot_subrange(100, 16).unwrap()),
        vec![0x22; 16]
    );
    // Full backing identity includes untouched pitch padding; GPU padding stores
    // must not invalidate native texels or fabricate a new presentation version.
    let (_, backing) = backing_allocation(4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(backing.clone(), description, planes)
        .unwrap();
    let version = cache.image(image).unwrap().version;
    let producer = cache.image(image).unwrap().producer;
    let pad = buffer(&mut cache, backing.snapshot_subrange(48, 8).unwrap());
    write(&mut cache, pad, 0, &[0x99; 8]);
    cache.flush().unwrap();
    cache.ensure_image(image).unwrap();
    assert_eq!(cache.image(image).unwrap().version, version);
    assert_eq!(cache.image(image).unwrap().producer, producer);
    let view = ImageViewDescription {
        format: vk::Format::R8G8B8A8_UNORM,
        view_type: vk::ImageViewType::TYPE_2D,
        aspects: vk::ImageAspectFlags::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
        swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
    };
    let old = cache.image_view(image, view).unwrap();
    cache.preserve_image(image).unwrap();
    while cache.evict_one().unwrap() {}
    assert!(cache.image_view(old.image, view).is_err());
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn packed_rgb565_keeps_native_storage_without_rgba_expansion() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x31);
    let description = NativeImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R5G6B5_UNORM_PACK16,
        extent: vk::Extent3D {
            width: 3,
            height: 2,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        flags: vk::ImageCreateFlags::empty(),
    };
    let plane = ImagePlane {
        offset: 2,
        row_pitch: 6,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent: description.extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 2,
    };
    let image = cache
        .image_for(backing.clone(), description, vec![plane])
        .unwrap();
    let (_, source) = backing_allocation(4096, 4096, 0x72);
    let source = buffer(&mut cache, source);
    cache.write_image_from_buffer(image, source, 0).unwrap();
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[2..14], &[0x72; 12]);
    assert_eq!(&bytes[..2], &[0x31; 2]);
    assert_eq!(bytes[14], 0x31);
    assert_eq!(cache.statistics().canonical_readback_bytes, 12);
    assert_eq!(
        cache.image(image).unwrap().description.format,
        vk::Format::R5G6B5_UNORM_PACK16
    );
    drop(cache);
    validation.check().unwrap();
}
