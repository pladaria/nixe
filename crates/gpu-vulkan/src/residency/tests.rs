use super::*;
use crate::{ApiPreference, QueueConfiguration, queue::QueueState};
use nixe_memory::{CanonicalAllocation, MemoryPermissions, VisibilityCoordinator};
use std::time::Duration;

#[test]
fn batched_cpu_streams_preserve_versions_aliases_and_native_imports() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(8192, 4096, 0x19);
    let a = buffer(&mut cache, backing.snapshot_subrange(64, 32).unwrap());
    let b = buffer(&mut cache, backing.snapshot_subrange(4096, 32).unwrap());
    let requests = [a, b].map(|id| {
        Some(buffer::BufferStreamRequest {
            id,
            offset: 0,
            size: 32,
            alignment: 16,
        })
    });
    let old = cache.stream_buffers(&requests).unwrap();
    allocation.write(64, &[0x37; 32]).unwrap();
    allocation.write(4096, &[0x42; 32]).unwrap();
    let new = cache.stream_buffers(&requests).unwrap();
    for (index, expected) in [0x37, 0x42].into_iter().enumerate() {
        let (buffer, offset) = new[index].as_ref().unwrap();
        let (previous, previous_offset) = old[index].as_ref().unwrap();
        // No GPU writes target these atom-isolated host upload ranges.
        assert_eq!(
            unsafe { buffer.read_range(*offset, 32) }.unwrap(),
            [expected; 32]
        );
        assert_eq!(
            unsafe { previous.read_range(*previous_offset, 32) }.unwrap(),
            [0x19; 32]
        );
    }
    assert_eq!(cache.read_buffer(a, 0, 32).unwrap(), [0x37; 32]);
    assert_eq!(cache.read_buffer(b, 0, 32).unwrap(), [0x42; 32]);
    write(&mut cache, a, 0, &[0x71; 32]);
    let streams = cache.stream_buffers(&requests).unwrap();
    assert!(
        streams[0].is_none(),
        "GPU-owned data must use its native producer"
    );
    assert!(streams[1].is_some());
    assert_eq!(cache.read_buffer(a, 0, 32).unwrap(), [0x71; 32]);
    drop(cache);
    validation.check().unwrap();
}

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
fn cpu_updates_survive_disjoint_gpu_writes_on_the_same_page() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    for (offset, size, submit) in [(64, 256, false), (3840, 412, false), (3840, 412, true)] {
        let (allocation, whole) = backing_allocation(8192, 4096, 0);
        let consumer = buffer(&mut cache, whole.snapshot_subrange(offset, size).unwrap());
        let producer = buffer(&mut cache, whole.snapshot_subrange(1024, 32).unwrap());
        allocation.write(offset as usize + 64, &[0x42; 64]).unwrap();
        write(&mut cache, producer, 0, &[0x71; 32]);
        if submit {
            cache.flush().unwrap();
        }
        assert_eq!(cache.read_buffer(consumer, 64, 64).unwrap(), [0x42; 64]);
        assert_eq!(cache.read_buffer(producer, 0, 32).unwrap(), [0x71; 32]);
    }
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn cpu_image_updates_survive_disjoint_gpu_writes_on_the_same_page() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(4096, 4096, 0);
    let (description, mut planes) = rgba_image();
    planes[0].offset = 128;
    let image = cache.image_for(whole.clone(), description, planes).unwrap();
    let producer = buffer(&mut cache, whole.snapshot_subrange(1024, 32).unwrap());
    allocation.write(128, &[0x42; 16]).unwrap();
    write(&mut cache, producer, 0, &[0x71; 32]);
    cache.ensure_image(image).unwrap();
    let (_, output) = backing_allocation(4096, 4096, 0);
    let destination = buffer(&mut cache, output.snapshot_subrange(0, 16).unwrap());
    cache
        .copy_source(Resource::Image(image), 128, destination, 0, 16)
        .unwrap();
    assert_eq!(cache.read_buffer(destination, 0, 16).unwrap(), [0x42; 16]);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
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
        &device.dispatch,
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
            .cpu_imports(&mixed, &dependency, &[(0, 128)])
            .unwrap()
            .snapshots,
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
            .cpu_imports(&mixed, &dependency, &[(110, 10), (4, 70)])
            .unwrap()
            .snapshots,
        vec![
            (0, updated.clone().into_boxed_slice()),
            (64, vec![0x33; 10].into_boxed_slice()),
            (96, updated.into_boxed_slice()),
        ]
    );
    assert!(
        cache
            .cpu_imports(&mixed, &dependency, &[])
            .unwrap()
            .snapshots
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
            .cpu_imports(&range, &dependency, &[(0, 1024)])
            .unwrap()
            .snapshots,
        vec![(0, expected.clone().into_boxed_slice())]
    );
    assert!(
        cache
            .cpu_imports(&range, &dependency, &[])
            .unwrap()
            .snapshots
            .is_empty()
    );
    let fragments = (0..128)
        .rev()
        .map(|index| (index * 8, 8))
        .collect::<Vec<_>>();
    assert_eq!(
        cache
            .cpu_imports(&range, &dependency, &fragments)
            .unwrap()
            .snapshots,
        vec![(0, expected.clone().into_boxed_slice())]
    );
    allocation.write(99, &[0xef; 5]).unwrap();
    expected[99..104].fill(0xef);
    assert_eq!(
        cache
            .cpu_imports(&range, &dependency, &[])
            .unwrap()
            .snapshots,
        vec![(0, expected.into_boxed_slice())]
    );
    assert!(
        cache
            .cpu_imports(&range, &dependency, &[])
            .unwrap()
            .snapshots
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
fn cpu_baseline_reuses_exact_revisions_and_outlives_canonical_acknowledgement() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, range) = backing_allocation(4096, 4096, 0x31);
    let id = buffer(&mut cache, range.clone());
    let page = range.pages().next().unwrap().identity();
    cache.prepare_write(Resource::Buffer(id), 12, 4).unwrap();
    let baseline = Arc::clone(&cache.visibility.pages.lock().unwrap()[&page].baseline);
    cache.prepare_write(Resource::Buffer(id), 16, 4).unwrap();
    assert!(
        Arc::ptr_eq(
            &baseline,
            &cache.visibility.pages.lock().unwrap()[&page].baseline
        ),
        "unchanged CPU input must not allocate/copy a new baseline"
    );
    write(&mut cache, id, 12, &[0x45; 4]);
    cache.flush().unwrap();
    assert!(Arc::ptr_eq(
        &baseline,
        &cache.visibility.pages.lock().unwrap()[&page].baseline
    ));
    let bytes = cpu_read(&mut cache, &range);
    assert_eq!(&bytes[12..16], &[0x45; 4]);
    cache.drain_notifications();
    assert!(
        cache.visibility.pages.lock().unwrap().contains_key(&page),
        "resident aliases still need the preserved baseline after CPU acknowledgement"
    );
    allocation.write(32, &[0x67; 4]).unwrap();
    cache.prepare_write(Resource::Buffer(id), 16, 4).unwrap();
    let changed = Arc::clone(&cache.visibility.pages.lock().unwrap()[&page].baseline);
    assert!(!Arc::ptr_eq(&baseline, &changed));
    assert_eq!(&changed[12..16], &[0x45; 4]);
    assert_eq!(&changed[32..36], &[0x67; 4]);
    write(&mut cache, id, 16, &[0x89; 4]);
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &range);
    assert_eq!(
        &bytes[12..20],
        &[0x45, 0x45, 0x45, 0x45, 0x89, 0x89, 0x89, 0x89]
    );
    assert_eq!(&bytes[32..36], &[0x67; 4]);
    drop(baseline);
    drop(changed);
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
    write(&mut cache, id, 11, &[0x15; 4]);
    cache.flush().unwrap();
    let version = || match range.pages().next().unwrap().visibility_state() {
        VisibilityState::GpuNewer { visible_at, .. } => visible_at,
        _ => panic!("device producer"),
    };
    let old = CpuVisibilityRequest {
        page: range.segments()[0].page(),
        size: 4096,
        device: NonCpuDeviceId::new(cache.instance.get()),
        visible_at: version(),
    };
    cache.materialize_cpu_page(old).unwrap();
    write(&mut cache, id, 12, &[0x25; 4]);
    cache.flush().unwrap();
    let newer = CpuVisibilityRequest {
        visible_at: version(),
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
fn preserved_baselines_retire_with_the_last_resident_alias() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, whole) = backing_allocation(4096, 4096, 0x19);
    let a = buffer(&mut cache, whole.snapshot_subrange(0, 16).unwrap());
    let _b = buffer(&mut cache, whole.snapshot_subrange(128, 16).unwrap());
    cache.prepare_write(Resource::Buffer(a), 0, 16).unwrap();
    assert_eq!(cache.visibility.pages.lock().unwrap().len(), 1);
    assert!(cache.evict_one().unwrap());
    assert_eq!(cache.visibility.pages.lock().unwrap().len(), 1);
    assert!(cache.evict_one().unwrap());
    assert!(cache.visibility.pages.lock().unwrap().is_empty());
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn large_cpu_uploads_progress_with_bounded_staging_and_preserve_gpu_prefixes() {
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
        &cache.device.dispatch,
    )
    .unwrap();
    let (allocation, range) = backing_allocation(4096, 4096, 0x34);
    let id = cache
        .buffer_for(range.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    assert!(cache.device.queue_usage().submitted_tick >= 7);
    assert_eq!(cache.device.queue_usage().overflow_capacity, 0);
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 4096);
    assert_eq!(cache.read_buffer(id, 0, 4096).unwrap(), vec![0x34; 4096]);
    allocation.write(100, &[0x56; 3]).unwrap();
    cache.ensure_buffer(id).unwrap();
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
fn dense_cpu_images_upload_once_without_buffers_and_preserve_later_gpu_writes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
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
        &cache.device.dispatch,
    )
    .unwrap();
    let (allocation, physical) = backing_allocation(16384, 256, 0x73);
    let backing = CanonicalBackingRange::mapped(
        physical.clone(),
        nixe_memory::CanonicalByteLayout {
            base: 0,
            columns: 57,
            rows: 33,
            element_bytes: 4,
            source_element_bytes: 4,
            source_byte_offset: 0,
            source_row_pitch: 228,
            tile: Some(
                nixe_memory::CanonicalByteTile::new(
                    64,
                    4,
                    (0..256).map(|i| (i % 64) * 4 + i / 64).collect(),
                )
                .unwrap(),
            ),
        },
    )
    .unwrap();
    let extent = vk::Extent3D {
        width: 57,
        height: 33,
        depth: 1,
    };
    let plane = ImagePlane {
        offset: 0,
        row_pitch: 228,
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let image = cache
        .image_for(
            backing.clone(),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![plane],
        )
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, backing.size());
    assert!(cache.buffers.iter().all(|slot| slot.entry.is_none()));
    assert!(backing.metadata_bytes() < backing.size() as usize);
    let (_, output) = backing_allocation(backing.size() as usize, 256, 0x23);
    let destination = cache
        .buffer_for_deferred_import(output, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    cache
        .prepare_write(Resource::Buffer(destination), 0, backing.size())
        .unwrap();
    cache
        .copy_source(Resource::Image(image), 0, destination, 0, backing.size())
        .unwrap();
    cache
        .record_authority(Resource::Buffer(destination), 0, backing.size())
        .unwrap();
    assert_eq!(
        cache.read_buffer(destination, 0, backing.size()).unwrap(),
        vec![0x73; backing.size() as usize]
    );
    cache
        .clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                float32: [0., 1., 0., 1.],
            },
        )
        .unwrap();
    cache.flush().unwrap();
    cpu_read(&mut cache, &physical);
    allocation.write(0, &[0x31; 16384]).unwrap();
    cache.ensure_image(image).unwrap();
    cache
        .clear_image_rect(
            image,
            0,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 1., 1.],
                },
            },
        )
        .unwrap();
    cache.flush().unwrap();
    let actual = cpu_read(&mut cache, &backing);
    assert_eq!(&actual[..4], &[0, 0, 255, 255]);
    assert!(
        actual[4..].iter().all(|&byte| byte == 0x31),
        "CPU replacements must retire previous GPU authority before a partial GPU writer"
    );
    assert!(cache.device.queue_usage().command_pools <= 2);
    assert_eq!(cache.device.queue_usage().overflow_capacity, 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn large_uploads_retain_streams_until_their_actual_consumer() {
    mod gate {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/gate.rs"
        ));
    }
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
        &cache.device.dispatch,
    )
    .unwrap();
    let (stream_slot, stream) = cache.device.submission.reserve_upload(&[0x42; 64]).unwrap();
    cache.device.submission.retain_stream_upload(stream_slot);
    let source = cache.device.submission.slots[stream_slot].upload_lease(stream);
    // Texture imports can flush repeatedly between preparing a uniform/vertex
    // stream and recording the draw that consumes it. Two slots and no overflow
    // force reuse; the stream's slot must remain unavailable throughout.
    let (_, range) = backing_allocation(4096, 4096, 0x34);
    let image_bytes = cache
        .buffer_for(range, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 4096);
    assert!(cache.device.queue_usage().submitted_tick >= 8);
    let (slot, destination, readback) = cache.device.submission.reserve_readback(64).unwrap();
    let command = cache.device.submission.slots[slot].command;
    let target = cache.device.submission.slots[slot]
        .readback_buffer(destination)
        .handle;
    unsafe {
        cache.device.dispatch.cmd_copy_buffer(
            command,
            source.handle,
            target,
            &[vk::BufferCopy::default()
                .src_offset(stream.offset)
                .dst_offset(destination.offset)
                .size(64)],
        );
    }
    recording::host_read_barrier(&cache.device, command, target, destination.offset, 64);
    cache.device.submission.release_stream_uploads();
    let mut gate = gate::Gate::new(&cache.device);
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let consumer = cache.flush_with_sync(&waits, &[]).unwrap().unwrap();
    gate.last_use = consumer.tick();
    assert!(
        !cache
            .device
            .wait_for_work(consumer, Duration::ZERO)
            .unwrap()
    );
    assert!(
        matches!(
            cache.device.submission.begin(),
            Err(VulkanError::QueuePressure { pending: Some(work), .. }) if work == consumer
        ),
        "the stream's upload submission has completed, but its actual consumer still owns the arena"
    );
    gate.signal();
    assert_eq!(cache.device.readback(readback).unwrap(), [0x42; 64]);
    assert_eq!(
        cache.read_buffer(image_bytes, 0, 4096).unwrap(),
        [0x34; 4096]
    );
    assert_eq!(cache.device.queue_usage().command_pools, 2);
    assert_eq!(cache.device.queue_usage().overflow_capacity, 0);
    drop(gate);
    drop(source);
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
fn asynchronous_page_versions_preserve_each_capture_and_never_replace_newer_cache_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x25);
    let id = buffer(&mut cache, backing.clone());
    let request = |backing: &CanonicalBackingRange| {
        let page = backing.pages().next().unwrap();
        let VisibilityState::GpuNewer { device, visible_at } = page.visibility_state() else {
            panic!("buffer producer")
        };
        CpuVisibilityRequest {
            page: page.identity(),
            size: page.size(),
            device,
            visible_at,
        }
    };
    write(&mut cache, id, 9, &[0xa7; 4]);
    cache.flush().unwrap();
    let first = request(&backing);
    assert!(cache.request_cpu_page(first).unwrap().is_none());
    // The first transfer precedes this real device writer in queue order.
    write(&mut cache, id, 9, &[0xb8; 4]);
    cache.flush().unwrap();
    let second = request(&backing);
    assert!(second.visible_at > first.visible_at);
    assert!(cache.request_cpu_page(second).unwrap().is_none());
    let work = cache.flush().unwrap().unwrap();
    cache
        .device
        .wait_for_work(work, Duration::from_secs(5))
        .unwrap();
    let newer = cache.request_cpu_page(second).unwrap().unwrap();
    let older = cache.request_cpu_page(first).unwrap().unwrap();
    assert_eq!(&newer[9..13], &[0xb8; 4]);
    assert_eq!(&older[9..13], &[0xa7; 4]);
    assert!(newer[..9].iter().chain(&newer[13..]).all(|&b| b == 0x25));
    let pages = cache.visibility.pages.lock().unwrap();
    let page = &pages[&first.page];
    assert_eq!(page.completed, Some(second.visible_at));
    assert_eq!(&page.bytes[9..13], &[0xb8; 4]);
    drop(pages);
    assert_eq!(&cpu_read(&mut cache, &backing)[9..13], &[0xb8; 4]);
    cache.drain_notifications();
    assert!(!cache.authoritative(Resource::Buffer(id)));
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
    assert_eq!(
        cache.visibility.pages.lock().unwrap().len(),
        1,
        "a resident alias retains its baseline independently of completed producer authority"
    );
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

#[test]
fn fragmented_image_import_batches_transfers_and_bounds_temporary_allocations() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (width, height) = (128_u32, 32_u32);
    let size = u64::from(width * height * 4);
    let (_, backing) = backing_allocation(size as usize, 4096, 0xa5);
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let plane = ImagePlane {
        offset: 0,
        row_pitch: u64::from(width * 4),
        mip_level: 0,
        array_layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        extent,
        block_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        bytes_per_block: 4,
    };
    let source = cache
        .image_for_device_write(
            backing.clone(),
            NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UINT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                flags: vk::ImageCreateFlags::empty(),
            },
            vec![plane],
        )
        .unwrap();
    cache
        .clear_color_image(
            source,
            0,
            vk::ClearColorValue {
                uint32: [255, 128, 64, 255],
            },
        )
        .unwrap();
    // Each logical byte comes from one green channel, producing 4096 separate
    // authoritative image intervals. Their native block copies are unaligned.
    let greens = CanonicalBackingRange::mapped(
        backing.clone(),
        nixe_memory::CanonicalByteLayout {
            base: 0,
            columns: width,
            rows: height,
            element_bytes: 1,
            source_element_bytes: 4,
            source_byte_offset: 1,
            source_row_pitch: u64::from(width * 4),
            tile: None,
        },
    )
    .unwrap();
    let before = cache.allocated_bytes();
    let buffer = cache
        .buffer_for(greens.clone(), vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap();
    assert!(backing.pages().all(|page| {
        cache
            .index
            .pages
            .get(&page.identity())
            .unwrap()
            .bindings_for(Resource::Buffer(buffer))
            .all(|spans| spans.is_compact())
    }));
    assert!(
        cache.allocated_bytes() - before < 64 * 1024,
        "fragment count must not allocate one native scratch buffer per byte"
    );
    assert_eq!(cache.statistics().native_readback_bytes, 0);
    let output = cache
        .read_buffer_native(buffer, 0, u64::from(width * height))
        .unwrap();
    assert!(output.iter().all(|&byte| byte == 128));
    // Small reconstructions retain a copy plan; subsequent refreshes must use
    // the same batched byte scatter rather than reintroducing per-byte buffers.
    let small = cache
        .buffer_for(
            greens.logical_view(0, 256).unwrap(),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    cache
        .clear_color_image(
            source,
            0,
            vk::ClearColorValue {
                uint32: [255, 160, 64, 255],
            },
        )
        .unwrap();
    let hits = cache.statistics().image_import_plan_hits;
    assert_eq!(cache.read_buffer(small, 0, 256).unwrap(), [160; 256]);
    assert!(cache.statistics().image_import_plan_hits > hits);
    let original = cpu_read(&mut cache, &backing);
    assert!(
        original
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 160, 64, 255])
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn pending_producers_and_cpu_downloads_share_one_submission() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x19);
    let id = buffer(&mut cache, backing.clone());
    write(&mut cache, id, 64, &[0x71; 32]);
    cache.accept_recording().unwrap();
    let before = cache.device.queue_usage().submitted_tick;
    let mut expected = vec![0x19; 4096];
    expected[64..96].fill(0x71);
    assert_eq!(cpu_read(&mut cache, &backing), expected);
    assert_eq!(
        cache.device.queue_usage().submitted_tick,
        before + 1,
        "pending producer and canonical download must submit together"
    );

    write(&mut cache, id, 96, &[0x93; 16]);
    let before = cache.device.queue_usage().submitted_tick;
    assert_eq!(cache.read_buffer(id, 96, 16).unwrap(), [0x93; 16]);
    assert_eq!(
        cache.device.queue_usage().submitted_tick,
        before + 1,
        "native diagnostic reads must not pre-submit their producer"
    );

    write(&mut cache, id, 128, &[0x55; 16]);
    let before = cache.device.queue_usage().submitted_tick;
    cache.preserve_buffer(id).unwrap();
    assert_eq!(&cpu_read(&mut cache, &backing)[128..144], &[0x55; 16]);
    assert_eq!(
        cache.device.queue_usage().submitted_tick,
        before + 1,
        "explicit preservation accepts pending ownership before recording its download"
    );

    let (_, backing) = backing_allocation(4096, 4096, 0x19);
    let (description, planes) = rgba_image();
    let plane = planes[0];
    let image = cache
        .image_for(backing.clone(), description, planes)
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
    cache.accept_recording().unwrap();
    let before = cache.device.queue_usage().submitted_tick;
    let actual = cpu_read(&mut cache, &backing);
    let mut expected = vec![0x19; 4096];
    for row in 0..plane.extent.height {
        let offset = (plane.offset + u64::from(row) * plane.row_pitch) as usize;
        for pixel in expected[offset..offset + plane.extent.width as usize * 4].chunks_exact_mut(4)
        {
            pixel.copy_from_slice(&[255, 0, 0, 255]);
        }
    }
    assert_eq!(
        actual, expected,
        "image downloads preserve baseline padding"
    );
    assert_eq!(
        cache.device.queue_usage().submitted_tick,
        before + 1,
        "pending clear and canonical image download must submit together"
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn image_cpu_imports_keep_another_alias_prepared_page_baseline() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    for observe_only in [false, true] {
        let (allocation, whole) = backing_allocation(4096, 4096, 0x19);
        let destination = buffer(&mut cache, whole.snapshot_subrange(0, 16).unwrap());
        let (mut description, mut planes) = rgba_image();
        description.extent.height = 1;
        planes[0].offset = 0;
        planes[0].row_pitch = 16;
        planes[0].extent.height = 1;
        let image = cache
            .image_for(
                whole.snapshot_subrange(128, 16).unwrap(),
                description,
                planes,
            )
            .unwrap();
        allocation.write(128, &[0x37; 16]).unwrap();
        // The write binding is prepared before importing a later read binding,
        // as in an actual draw/dispatch that aliases one physical CPU page.
        cache
            .prepare_write(Resource::Buffer(destination), 0, 4)
            .unwrap();
        if observe_only {
            cache.observe_image_cpu_writes(image).unwrap();
        } else {
            cache.ensure_image(image).unwrap();
        }
        cache
            .upload_buffer(destination, 0, &0x11223344_u32.to_le_bytes())
            .unwrap();
        cache
            .record_authority(Resource::Buffer(destination), 0, 4)
            .unwrap();
        cache.flush().unwrap();
        let result = cpu_read(&mut cache, &whole);
        assert_eq!(&result[..4], &0x11223344_u32.to_le_bytes());
        assert_eq!(&result[4..128], &[0x19; 124]);
        assert_eq!(&result[128..144], &[0x37; 16]);
        assert!(result[144..].iter().all(|&b| b == 0x19));
    }
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn stronger_uniform_stream_alignment_never_reuses_a_misaligned_capture() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let monitor = cache.device.validation_monitor();
    let (_, backing) = backing_allocation(4096, 4096, 0x39);
    let padding = buffer(&mut cache, backing.snapshot_subrange(0, 4).unwrap());
    let input = buffer(&mut cache, backing.snapshot_subrange(64, 32).unwrap());
    let requests = [
        Some(buffer::BufferStreamRequest {
            id: padding,
            offset: 0,
            size: 4,
            alignment: 4,
        }),
        Some(buffer::BufferStreamRequest {
            id: input,
            offset: 0,
            size: 32,
            alignment: 4,
        }),
    ];
    let captured = cache.capture_buffer_inputs(&requests).unwrap();
    let strong = [Some(buffer::BufferStreamRequest {
        alignment: 4096,
        ..requests[1].unwrap()
    })];
    let streams = cache
        .stream_buffers_with_inputs(&strong, Some(&captured))
        .unwrap();
    let (buffer, offset) = streams[0].as_ref().unwrap();
    assert_eq!(*offset % 4096, 0);
    assert!(buffer.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
    assert_eq!(
        unsafe { buffer.read_range(*offset, 32) }.unwrap(),
        [0x39; 32]
    );
    drop(captured);
    drop(streams);
    drop(cache);
    monitor.check().unwrap();
}
