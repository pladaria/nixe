use super::*;
use crate::ApiPreference;
use nixe_memory::{Allocation, AllocationRange, Machine, PAGE_SIZE};
use std::time::Duration;

#[test]
fn batched_cpu_streams_preserve_versions_aliases_and_native_imports() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(&cache, 8192, 4096, 0x19);
    let a = buffer(&mut cache, backing.slice(64, 32).unwrap());
    let b = buffer(&mut cache, backing.slice(4096, 32).unwrap());
    let requests = [(a, 64), (b, 0)].map(|(id, offset)| {
        Some(buffer::BufferStreamRequest {
            id,
            offset,
            size: 32,
            alignment: 16,
        })
    });
    let old = requests
        .iter()
        .map(|request| cache.stream_buffer(request.unwrap()).unwrap())
        .collect::<Vec<_>>();
    crate::residency::tests::cpu_write(&mut cache, &allocation, 64, &[0x37; 32]).unwrap();
    crate::residency::tests::cpu_write(&mut cache, &allocation, 4096, &[0x42; 32]).unwrap();
    let new = requests
        .iter()
        .map(|request| cache.stream_buffer(request.unwrap()).unwrap())
        .collect::<Vec<_>>();
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
    assert_eq!(cache.read_buffer(a, 64, 32).unwrap(), [0x37; 32]);
    assert_eq!(cache.read_buffer(b, 0, 32).unwrap(), [0x42; 32]);
    write(&mut cache, a, 64, &[0x71; 32]);
    let streams = requests
        .iter()
        .map(|request| cache.stream_buffer(request.unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(
        streams[0].is_none(),
        "GPU-owned data must use its native producer"
    );
    assert!(streams[1].is_some());
    assert_eq!(cache.read_buffer(a, 64, 32).unwrap(), [0x71; 32]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn cpu_streaming_buffer_imports_keep_independent_aliases_and_gpu_handoffs() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
    let a = buffer(&mut cache, backing.clone());
    let b = cache
        .buffer_for(backing.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    for value in 0..16 {
        crate::residency::tests::cpu_write(&mut cache, &allocation, 17, &[value; 37]).unwrap();
        assert_eq!(cache.read_buffer(a, 17, 37).unwrap(), [value; 37]);
        assert_eq!(cache.read_buffer(b, 17, 37).unwrap(), [value; 37]);
    }
    write(&mut cache, a, 17, &[0xa7; 37]);
    assert_eq!(cache.read_buffer(b, 17, 37).unwrap(), [0xa7; 37]);
    assert_eq!(&cpu_read(&mut cache, &backing)[17..54], &[0xa7; 37]);
    crate::residency::tests::cpu_write(&mut cache, &allocation, 20, &[0x42; 7]).unwrap();
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
        let (allocation, whole) = backing_allocation(&cache, 8192, 4096, 0);
        let consumer = buffer(&mut cache, whole.slice(offset, size).unwrap());
        let producer = buffer(&mut cache, whole.slice(1024, 32).unwrap());
        crate::residency::tests::cpu_write(
            &mut cache,
            &allocation,
            offset as usize + 64,
            &[0x42; 64],
        )
        .unwrap();
        write(&mut cache, producer, 1024, &[0x71; 32]);
        if submit {
            cache.flush().unwrap();
        }
        assert_eq!(
            cache.read_buffer(consumer, offset + 64, 64).unwrap(),
            [0x42; 64]
        );
        assert_eq!(cache.read_buffer(producer, 1024, 32).unwrap(), [0x71; 32]);
    }
    // Fence publication can retire the three exact 32-byte writes, never
    // the unrelated bytes of their shared page.
    assert!(cache.statistics().canonical_readback_bytes <= 3 * 32);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn cpu_image_updates_survive_disjoint_gpu_writes_on_the_same_page() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(&cache, 4096, 4096, 0);
    let (description, mut planes) = rgba_image();
    planes[0].offset = 128;
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(whole.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    let producer = buffer(&mut cache, whole.slice(1024, 32).unwrap());
    crate::residency::tests::cpu_write(&mut cache, &allocation, 128, &[0x42; 16]).unwrap();
    write(&mut cache, producer, 1024, &[0x71; 32]);
    cache.ensure_image(image).unwrap();
    let (_allocation_1, output) = backing_allocation(&cache, 4096, 4096, 0);
    let destination = buffer(&mut cache, output.slice(0, 16).unwrap());
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
    let (allocation, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
    let prepared = cache
        .buffer_for_deferred_import(backing.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 0);
    crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &[0x37; 4096]).unwrap();
    assert_eq!(cache.read_buffer(prepared, 0, 4096).unwrap(), [0x37; 4096]);
    let producer = buffer(&mut cache, backing.clone());
    write(&mut cache, producer, 0, &[0x51; 4096]);
    let before = cache.statistics().cpu_uploaded_bytes;
    assert_eq!(
        cache
            .buffer_for_deferred_import(backing, vk::BufferUsageFlags::VERTEX_BUFFER)
            .unwrap()
            .buffer,
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
    let (allocation, whole) = backing_allocation(&cache, 4096, 4096, 0x19);
    let mut ids = Vec::new();
    for offset in (0..2048).step_by(16) {
        ids.push(buffer(&mut cache, whole.slice(offset, 16).unwrap()));
    }
    let readonly = whole.clone();
    for (index, &id) in ids.iter().enumerate() {
        assert_eq!(
            buffer(&mut cache, readonly.slice(index as u64 * 16, 16).unwrap()),
            id
        );
    }
    let range = whole.slice(16, 16).unwrap();
    let vertex = cache
        .buffer_for(range.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap()
        .buffer;
    assert_eq!(vertex, ids[1]);
    assert_eq!(buffer(&mut cache, range.clone()), ids[1]);
    crate::residency::tests::cpu_write(&mut cache, &allocation, 16, &[0x37; 16]).unwrap();
    assert_eq!(cache.read_buffer(ids[1], 16, 16).unwrap(), [0x37; 16]);
    assert_eq!(cache.read_buffer(vertex, 16, 16).unwrap(), [0x37; 16]);
    cache.flush().unwrap();
    if let Some(work) = cache.flush().unwrap() {
        cache.device.wait_for_work(work, Duration::MAX).unwrap();
    }
    cache.device.poll_progress().unwrap();
    while cache.evict_one().unwrap() {}
    assert!(cache.buffer(ids[1]).is_err());
    let next = buffer(&mut cache, range);
    assert_ne!(next, ids[1]);
    assert_eq!(cache.read_buffer(next, 16, 16).unwrap(), [0x37; 16]);
    drop(cache);
    validation.check().unwrap();
}

use crate::test_device as support;
mod gate {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/gate.rs"
    ));
}
pub(crate) fn cache(api: ApiPreference, budget: u64) -> Option<VulkanResidency> {
    let device = support::device_for_test(api)?;

    Some(VulkanResidency::new(Machine::new(64 * 1024 * 1024).unwrap(), device, budget).unwrap())
}
pub(crate) fn backing_allocation(
    cache: &VulkanResidency,
    size: usize,
    _page_size: usize,
    byte: u8,
) -> (Allocation, Arc<PhysicalMapping>) {
    let machine = &cache.memory;
    let bytes = (size as u64).next_multiple_of(PAGE_SIZE);
    let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, size as u64).unwrap();
    machine
        .write_resolved(std::slice::from_ref(&range), &vec![byte; size])
        .unwrap();
    let mapping = PhysicalMapping::new(&[range]).unwrap();
    (allocation, mapping)
}
pub(crate) fn cpu_read(cache: &mut VulkanResidency, range: &PhysicalMapping) -> Vec<u8> {
    let inputs = range.pin(0, range.bytes()).unwrap();
    let nixe_memory::AccessResult::Read(bytes) = cache
        .access_memory(inputs, nixe_memory::AccessOperation::Read)
        .unwrap()
    else {
        panic!("expected read")
    };
    bytes.into_vec()
}
pub(super) fn buffer(cache: &mut VulkanResidency, range: Arc<PhysicalMapping>) -> BufferId {
    cache
        .buffer_for(range, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap()
        .buffer
}
pub(crate) fn image_geometry(
    mapping: Arc<PhysicalMapping>,
    description: NativeImageDescription,
    planes: &[ImagePlane],
) -> Vec<Arc<Footprint>> {
    use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain};
    planes
        .iter()
        .map(|plane| {
            let grid = match description.samples {
                vk::SampleCountFlags::TYPE_1 => [1, 1],
                vk::SampleCountFlags::TYPE_4 => [2, 2],
                _ => panic!("fixture sample grid must be explicit"),
            };
            let width = (plane.extent.width * grid[0]).div_ceil(plane.block_extent.width);
            let rows = (plane.extent.height * grid[1]).div_ceil(plane.block_extent.height);
            let aspect_bytes = if plane.aspect == vk::ImageAspectFlags::DEPTH
                && matches!(
                    description.format,
                    vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32
                ) {
                3
            } else {
                plane.bytes_per_block
            };
            Footprint::new(
                mapping.clone(),
                WriteDomain::Image(ImageFootprint {
                    depth_slice: 0,
                    base: plane.offset,
                    columns: width,
                    rows,
                    packed_bytes: plane.bytes_per_block,
                    aspect_offset: 0,
                    aspect_bytes,
                    storage: ImageStorage::Pitch {
                        row_pitch: plane.row_pitch,
                    },
                    native_base: plane.offset,
                    native_element_bytes: plane.bytes_per_block,
                    native_row_pitch: plane.row_pitch,
                    rect: ByteRect {
                        x: 0,
                        y: 0,
                        width,
                        height: rows,
                    },
                }),
            )
            .unwrap()
        })
        .collect()
}
pub(crate) fn plane_geometry(
    mapping: Arc<PhysicalMapping>,
    extent: [u32; 2],
    row_pitch: u64,
    bytes: u32,
) -> Arc<Footprint> {
    use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain};
    Footprint::new(
        mapping,
        WriteDomain::Image(ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: extent[0],
            rows: extent[1],
            packed_bytes: bytes,
            aspect_offset: 0,
            aspect_bytes: bytes,
            storage: ImageStorage::Pitch { row_pitch },
            native_base: 0,
            native_element_bytes: bytes,
            native_row_pitch: row_pitch,
            rect: ByteRect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
        }),
    )
    .unwrap()
}
pub(crate) fn footprint_bytes(footprint: &Footprint) -> u64 {
    let nixe_gpu::physical::WriteDomain::Image(domain) = footprint.domain() else {
        panic!("image fixture")
    };
    u64::from(domain.rect.width)
        * u64::from(domain.rect.height)
        * u64::from(domain.native_element_bytes)
}
pub(crate) fn footprint_buffer(cache: &mut VulkanResidency, footprint: Arc<Footprint>) -> BufferId {
    let mut parts = Vec::new();
    for (index, span) in footprint.mapping().spans().iter().enumerate() {
        footprint
            .visit(index, span.start, span.bytes, |part| {
                parts.push((part.native, index, part.start, part.bytes));
                true
            })
            .unwrap();
    }
    parts.sort_unstable_by_key(|part| part.0);
    let ranges: Vec<_> = parts
        .into_iter()
        .map(|(_, index, start, bytes)| {
            footprint
                .mapping()
                .pin_span(index, start, bytes)
                .unwrap()
                .unwrap()
        })
        .collect();
    cache
        .buffer_for(
            PhysicalMapping::new(&ranges).unwrap(),
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .buffer
}
/// Establish image producers using the same native image copy used by guest blits.
fn copy_fixture_image(
    cache: &mut VulkanResidency,
    destination: ImageId,
    mapping: Arc<PhysicalMapping>,
) {
    let entry = cache.image(destination).unwrap();
    let description = entry.description;
    let planes = entry.planes.to_vec();
    let source = cache
        .image_for(
            image_geometry(mapping, description, &planes),
            description,
            planes.clone(),
        )
        .unwrap();
    for (index, plane) in planes.into_iter().enumerate() {
        cache
            .copy_image_region(
                source,
                index,
                transfers::full_rect(plane),
                destination,
                index,
                vk::Offset2D::default(),
            )
            .unwrap();
    }
}
pub(crate) fn footprint_read(cache: &mut VulkanResidency, footprint: Arc<Footprint>) -> Vec<u8> {
    let bytes = footprint_bytes(&footprint);
    let buffer = footprint_buffer(cache, footprint);
    cache.read_buffer(buffer, 0, bytes).unwrap()
}
pub(crate) fn cpu_write(
    cache: &mut VulkanResidency,
    allocation: &Allocation,
    offset: impl TryInto<u64>,
    bytes: &[u8],
) -> Result<(), VulkanError> {
    let offset = offset.try_into().ok().expect("fixture offset");
    let input = AllocationRange::new(allocation.pin(), offset, bytes.len() as u64)?;
    let result = cache.access_memory(
        vec![input],
        nixe_memory::AccessOperation::Write(bytes.into()),
    )?;
    assert!(matches!(result, nixe_memory::AccessResult::Written));
    let memory = cache.memory().clone();
    cache.consume_cpu_writes(memory.collect_cpu_writes()?)?;
    Ok(())
}
fn write(cache: &mut VulkanResidency, id: BufferId, offset: u64, bytes: &[u8]) -> WorkTicket {
    cache.write_buffer(id, offset, bytes).unwrap()
}
#[test]
fn a_batch_of_fragmented_cpu_overwrites_preserves_native_buffer_neighbors() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let (allocation, whole) = backing_allocation(&cache, 4096, 4096, 0x19);
    let id = buffer(&mut cache, whole);
    write(&mut cache, id, 0, &[11; 4096]);
    let memory = cache.memory().clone();
    let space =
        nixe_memory::AddressSpace::new(memory.clone(), nixe_memory::AddressWidth::Bits32).unwrap();
    space
        .map(
            nixe_memory::GuestVirtualAddress::new(4096),
            allocation.clone(),
            0,
            4096,
            nixe_memory::MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    memory.enable_cpu_write_collection().unwrap();
    // The first write invalidates the upload's CPU observation on the owned
    // access path, as a deferred native store does. Later unobserved stores batch.
    cpu_write(&mut cache, &allocation, 3u64, &[77]).unwrap();
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    let mut expected = vec![11; 4096];
    expected[3] = 77;
    for offset in (20..4096).step_by(17) {
        assert!(
            space
                .write_native(
                    &native,
                    nixe_memory::GuestVirtualAddress::new(4096 + offset as u64),
                    &[77]
                )
                .unwrap()
        );
        expected[offset] = 77;
    }
    drop(native);
    cache
        .consume_cpu_writes(memory.collect_cpu_writes().unwrap())
        .unwrap();
    assert_eq!(cache.read_buffer(id, 0, 4096).unwrap(), expected);
    memory.disable_cpu_write_collection();
}

fn exercise_buffer_coherence(api: ApiPreference) {
    let Some(mut cache) = cache(api, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(&cache, 3 * 4096, 4096, 0x19);
    let a = buffer(&mut cache, whole.clone());
    assert_eq!(cache.statistics().cpu_uploaded_bytes, whole.bytes());
    assert_eq!(
        cache.device.queue_usage().staging_capacity,
        0,
        "stream uploads create no pooled readback buffers"
    );
    let old = cache.statistics();
    for _ in 0..100 {
        assert_eq!(buffer(&mut cache, whole.clone()), a);
    }
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        old.cpu_uploaded_bytes
    );
    crate::residency::tests::cpu_write(&mut cache, &allocation, 4096 + 100, &[0x63; 23]).unwrap();
    cache.ensure_buffer(a).unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes - old.cpu_uploaded_bytes,
        4096
    );
    let before_downloads = cache.statistics().native_readback_bytes;
    write(&mut cache, a, 101, &[0xaa; 7]);
    cache.flush().unwrap();
    let b = buffer(&mut cache, whole.slice(80, 80).unwrap());
    assert_eq!(
        cache.statistics().canonical_readback_bytes,
        0,
        "new alias must use GPU producers"
    );
    let data = cache.read_buffer(b, 80, 80).unwrap();
    let mut expected = vec![0x19; 80];
    expected[21..28].fill(0xaa);
    assert_eq!(data, expected);
    // Independent producer on the same conservative page.
    write(&mut cache, b, 130, &[0xbb; 9]);
    cache.flush().unwrap();
    let data = cpu_read(&mut cache, &whole.slice(0, 4096).unwrap());
    assert_eq!(&data[101..108], &[0xaa; 7]);
    assert_eq!(&data[130..139], &[0xbb; 9]);
    assert!(data[..101].iter().all(|byte| *byte == 0x19));
    assert_eq!(
        cache.statistics().native_readback_bytes - before_downloads,
        80 + 7 + 9,
        "the diagnostic read and two exact producer downloads each transfer once"
    );
    assert_eq!(cache.statistics().canonical_readback_bytes, 16);
    crate::residency::tests::cpu_write(&mut cache, &allocation, 132, &[0xdd; 2]).unwrap();
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
fn fragmented_repeated_canonical_offsets_and_same_buffer_copies() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (a, range_a) = backing_allocation(&cache, 4096, 4096, 0x31);
    let (_allocation_6, range_b) = backing_allocation(&cache, 4096, 4096, 0x74);
    let segments = [
        range_a.slice(100, 32).unwrap(),
        range_b.slice(200, 32).unwrap(),
        range_a.slice(100, 32).unwrap(),
    ];
    let range = PhysicalMapping::join(&segments).unwrap();
    let id = buffer(&mut cache, range.clone());
    crate::residency::tests::cpu_write(&mut cache, &a, 105, &[0x55; 4]).unwrap();
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
    assert_eq!(
        cpu_read(&mut cache, &range_a.slice(100, 32).unwrap()),
        vec![0x22; 32]
    );
    cache
        .copy_physical_buffers(
            &range,
            &range,
            &[nixe_gpu::native::BufferCopyRegion {
                source: 0,
                destination: 1,
                bytes: 12,
            }],
        )
        .unwrap();
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
            depth_slice: 0,
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
    let (_allocation_7, backing) = backing_allocation(&cache, 4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes.clone()),
            description,
            planes.clone(),
        )
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
    let alias = buffer(&mut cache, backing.slice(0, 128).unwrap());
    let mut other_description = description;
    other_description.format = vk::Format::B8G8R8A8_UNORM;
    let other = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), other_description, &planes),
            other_description,
            planes,
        )
        .unwrap();
    for color in [[0., 1., 0., 1.], [0., 0., 1., 1.], [1., 1., 1., 1.]] {
        let version = cache.image(image).unwrap().version;
        cache
            .clear_color_image(image, 0, vk::ClearColorValue { float32: color })
            .unwrap();
        assert_eq!(cache.image(image).unwrap().version, version + 1);
        assert!(
            cache
                .buffer(alias)
                .unwrap()
                .contents
                .device
                .iter()
                .any(ResourceObservation::is_invalid)
        );
        assert!(
            cache
                .image(other)
                .unwrap()
                .contents
                .device
                .iter()
                .any(ResourceObservation::is_invalid)
        );
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
            cpu_read(&mut cache, &backing.slice(0, 128).unwrap()),
            expected
        );
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
    let (allocation, backing) = backing_allocation(&cache, 4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes.clone()),
            description,
            planes.clone(),
        )
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
    let (_allocation_8, source_backing) = backing_allocation(&cache, 4096, 4096, 0x87);
    copy_fixture_image(&mut cache, image, source_backing);
    cache.flush().unwrap();
    let alias = buffer(&mut cache, backing.slice(30, 70).unwrap());
    let bytes = cache.read_buffer(alias, 30, 70).unwrap();
    let mut expected = vec![0x41; 70];
    for row in 0..3 {
        expected[2 + row * 24..18 + row * 24].fill(0x87);
    }
    assert_eq!(bytes, expected);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    write(&mut cache, alias, 40, &[0xa3; 5]);
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
    crate::residency::tests::cpu_write(&mut cache, &allocation, 33, &[0xf1; 3]).unwrap();
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cache
            .image_for(
                crate::residency::tests::image_geometry(backing, description, &planes),
                description,
                planes
            )
            .unwrap(),
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
    let (_allocation_9, range) = backing_allocation(&cache, 4096, 4096, 0x29);
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
            .flush(cache.device.queue, &mut cache.device.dispatch, &waits, &[])
    }
    .unwrap()
    .unwrap();
    gate.last_use = work.tick();
    assert!(!cache.evict_one().unwrap());
    assert_eq!(
        cache.allocated_bytes(),
        charged,
        "retired in-flight allocation remains charged"
    );
    assert!(cache.ensure_buffer(old).is_ok());
    gate.signal();
    cache
        .device
        .wait_for_work(work, Duration::from_secs(5))
        .unwrap();
    assert!(cache.evict_one().unwrap());
    assert!(cache.ensure_buffer(old).is_err());
    assert_eq!(cache.allocated_bytes(), 0);
    let new = buffer(&mut cache, range.clone());
    assert_ne!(new, old);
    write(&mut cache, new, 11, &[0x88; 5]);
    let work = cache.flush().unwrap().unwrap();
    cache.device.wait_for_work(work, Duration::MAX).unwrap();
    assert!(
        cache.evict_one().unwrap(),
        "authoritative eviction preserves newest bytes"
    );
    assert_eq!(&cpu_read(&mut cache, &range)[11..16], &[0x88; 5]);
    assert!(cache.ensure_buffer(new).is_err());
    assert_eq!(cache.statistics().canonical_readback_bytes, 5);
    drop(gate);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn dense_cpu_images_upload_once_without_buffers_and_preserve_later_gpu_writes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();

    let (allocation, physical) = backing_allocation(&cache, 16384, 256, 0x73);
    let backing = Footprint::new(
        physical.clone(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: 57,
            rows: 33,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: nixe_gpu::physical::ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
            },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 57_u64 * 4_u64,
            rect: nixe_gpu::physical::ByteRect {
                x: 0,
                y: 0,
                width: 57,
                height: 33,
            },
        }),
    )
    .unwrap();
    let extent = vk::Extent3D {
        width: 57,
        height: 33,
        depth: 1,
    };
    let plane = ImagePlane {
        depth_slice: 0,
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
            vec![backing.clone()],
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
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        footprint_bytes(&backing)
    );
    assert!(cache.buffers.iter().all(|slot| slot.entry.is_none()));
    assert!(backing.mapping().metadata_bytes() < footprint_bytes(&backing) as usize);
    assert_eq!(
        footprint_read(&mut cache, backing.clone()),
        vec![0x73; footprint_bytes(&backing) as usize]
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
    crate::residency::tests::cpu_write(&mut cache, &allocation, 0, &[0x31; 16384]).unwrap();
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
    let actual = footprint_read(&mut cache, backing.clone());
    assert_eq!(&actual[..4], &[0, 0, 255, 255]);
    assert!(
        actual[4..].iter().all(|&byte| byte == 0x31),
        "CPU replacements must retire previous GPU authority before a partial GPU writer"
    );
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

    let stream = cache
        .device
        .submission
        .reserve_upload(&mut cache.device.dispatch, &[0x42; 64])
        .unwrap();
    cache.device.submission.retain_stream_upload(&stream);
    let source = Arc::clone(cache.device.submission.staging_buffer(&stream));
    // Texture imports can flush repeatedly between preparing a uniform/vertex
    // stream and recording the draw that consumes it. Native contexts may
    // recycle while the actual consumer still retains these upload bytes.
    let (_allocation_12, range) = backing_allocation(&cache, 4096, 4096, 0x34);
    let image_bytes = cache
        .buffer_for(range, vk::BufferUsageFlags::TRANSFER_SRC)
        .unwrap()
        .buffer;
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 4096);
    let uploaded = cache.flush().unwrap().unwrap();
    cache
        .device
        .wait_for_work(uploaded, Duration::from_secs(5))
        .unwrap();
    let (destination, readback) = cache
        .device
        .submission
        .reserve_readback(&mut cache.device.dispatch, 64)
        .unwrap();

    let target = cache.device.submission.staging_buffer(&destination).handle;
    unsafe {
        cache.device.dispatch.cmd_copy_buffer(
            source.handle,
            target,
            &[vk::BufferCopy::default()
                .src_offset(stream.offset)
                .dst_offset(destination.offset)
                .size(64)],
        );
    }
    recording::host_read_barrier(&mut cache.device, target, destination.offset, 64);
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
    // Grow/reuse native contexts without touching the still-active mapped
    // stream region. Wrapping must fall back instead of overwriting it.
    cache.device.begin_recording().unwrap();
    let mut last = None;
    for _ in 0..16 {
        last = Some(
            cache
                .device
                .submission
                .reserve_slice(
                    &mut cache.device.dispatch,
                    crate::queue::staging::REGION_BYTES,
                    true,
                    256,
                )
                .unwrap(),
        );
    }
    let last = last.unwrap();
    assert!(!cache.device.submission.staging.is_stream(&last));
    let buffer = cache.device.submission.staging_buffer(&last);
    unsafe {
        buffer.write_range(last.offset, &[0x99; 64]).unwrap();
    }
    gate.signal();
    assert_eq!(cache.device.readback(readback).unwrap(), [0x42; 64]);
    assert_eq!(
        cache.read_buffer(image_bytes, 0, 4096).unwrap(),
        [0x34; 4096]
    );
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
    let (_allocation_13, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
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
                depth_slice: 0,
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
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes.clone()),
            description,
            planes.clone(),
        )
        .unwrap();
    let (_allocation_14, source) = backing_allocation(&cache, 4096, 4096, 0x71);
    copy_fixture_image(&mut cache, image, source);
    cache.flush().unwrap();
    // Splitting one BC block requests exactly three bytes, realigned on GPU.
    let alias = buffer(&mut cache, backing.slice(3, 3).unwrap());
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
fn repeated_image_rows_and_gpu_padding_writes_keep_texel_identity() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_15, original) = backing_allocation(&cache, 4096, 4096, 0x31);
    let part = original.slice(100, 16).unwrap();
    let repeated = PhysicalMapping::join(&[part.clone(), part]).unwrap();
    let (mut description, mut planes) = rgba_image();
    description.extent.height = 2;
    planes[0].offset = 0;
    planes[0].row_pitch = 16;
    planes[0].extent.height = 2;
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(repeated, description, &planes),
            description,
            planes,
        )
        .unwrap();
    let (source_allocation, source_backing) = backing_allocation(&cache, 4096, 4096, 0x11);
    crate::residency::tests::cpu_write(&mut cache, &source_allocation, 16, &[0x22; 16]).unwrap();
    copy_fixture_image(&mut cache, image, source_backing);
    cache.flush().unwrap();
    assert!(
        cache
            .image(image)
            .unwrap()
            .contents
            .device
            .iter()
            .any(ResourceObservation::is_invalid),
        "the earlier physical alias must observe the later row producer"
    );
    cache.ensure_image(image).unwrap();
    assert_eq!(
        cpu_read(&mut cache, &original.slice(100, 16).unwrap()),
        vec![0x22; 16]
    );
    // Full backing identity includes untouched pitch padding; GPU padding stores
    // must not invalidate native texels or fabricate a new presentation version.
    let (_allocation_16, backing) = backing_allocation(&cache, 4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    cache.ensure_image(image).unwrap();
    let version = cache.image(image).unwrap().version;
    let producer = cache.image(image).unwrap().producer;
    let pad = buffer(&mut cache, backing.slice(48, 8).unwrap());
    write(&mut cache, pad, 48, &[0x99; 8]);
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
    assert_eq!(cache.image_view(image, view).unwrap().image, image);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn packed_rgb565_keeps_native_storage_without_rgba_expansion() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation_17, backing) = backing_allocation(&cache, 4096, 4096, 0x31);
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
        depth_slice: 0,
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
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &[plane]),
            description,
            vec![plane],
        )
        .unwrap();
    let (_allocation_18, source) = backing_allocation(&cache, 4096, 4096, 0x72);
    copy_fixture_image(&mut cache, image, source);
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
    let (_allocation_19, backing) = backing_allocation(&cache, size as usize, 4096, 0xa5);
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let plane = ImagePlane {
        depth_slice: 0,
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
            crate::residency::tests::image_geometry(
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
                &[plane],
            ),
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
    let greens = Footprint::new(
        backing.clone(),
        nixe_gpu::physical::WriteDomain::Image(nixe_gpu::physical::ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: width,
            rows: height,
            packed_bytes: 4,
            aspect_offset: 1,
            aspect_bytes: 1,
            storage: nixe_gpu::physical::ImageStorage::Pitch {
                row_pitch: u64::from(width * 4),
            },
            native_base: 0,
            native_element_bytes: 1,
            native_row_pitch: u64::from(width),
            rect: nixe_gpu::physical::ByteRect {
                x: 0,
                y: 0,
                width,
                height,
            },
        }),
    )
    .unwrap();
    let before = cache.allocated_bytes();
    let buffer = footprint_buffer(&mut cache, greens.clone());
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
    let nixe_gpu::physical::WriteDomain::Image(mut subset) = greens.domain() else {
        unreachable!()
    };
    subset.rect.height = 2;
    let small = footprint_buffer(
        &mut cache,
        Footprint::new(
            greens.mapping().clone(),
            nixe_gpu::physical::WriteDomain::Image(subset),
        )
        .unwrap(),
    );
    cache
        .clear_color_image(
            source,
            0,
            vk::ClearColorValue {
                uint32: [255, 160, 64, 255],
            },
        )
        .unwrap();
    assert_eq!(cache.read_buffer(small, 0, 256).unwrap(), [160; 256]);
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
    let (_allocation_20, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
    let id = buffer(&mut cache, backing.clone());
    write(&mut cache, id, 64, &[0x71; 32]);
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
    assert_eq!(&cpu_read(&mut cache, &backing)[128..144], &[0x55; 16]);
    assert_eq!(
        cache.device.queue_usage().submitted_tick,
        before + 1,
        "CPU reads accept pending ownership before recording their download"
    );

    let (_allocation_21, backing) = backing_allocation(&cache, 4096, 4096, 0x19);
    let (description, planes) = rgba_image();
    let plane = planes[0];
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes),
            description,
            planes,
        )
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
fn stronger_uniform_stream_alignment_never_reuses_a_misaligned_capture() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let monitor = cache.device.validation_monitor();
    let (_allocation_22, backing) = backing_allocation(&cache, 4096, 4096, 0x39);
    let padding = buffer(&mut cache, backing.slice(0, 4).unwrap());
    let input = buffer(&mut cache, backing.slice(64, 32).unwrap());
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
    let _streams = requests
        .iter()
        .map(|request| cache.stream_buffer(request.unwrap()).unwrap())
        .collect::<Vec<_>>();
    let strong = [Some(buffer::BufferStreamRequest {
        alignment: 4096,
        ..requests[1].unwrap()
    })];
    let streams = strong
        .iter()
        .map(|request| cache.stream_buffer(request.unwrap()).unwrap())
        .collect::<Vec<_>>();
    let (buffer, offset) = streams[0].as_ref().unwrap();
    assert_eq!(*offset % 4096, 0);
    assert!(buffer.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
    assert_eq!(
        unsafe { buffer.read_range(*offset, 32) }.unwrap(),
        [0x39; 32]
    );
    drop(streams);
    drop(cache);
    monitor.check().unwrap();
}

#[test]
fn query_reports_commit_at_the_next_fence_and_keep_cpu_overwrites() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(&cache, 4096, 4096, 0x19);
    let ranges = |offset| vec![AllocationRange::new(allocation.pin(), offset, 4).unwrap()];
    cache
        .report_memory(ranges(64), 0, &Arc::from([0x37; 4]), false)
        .unwrap();
    cache
        .report_memory(ranges(128), 0, &Arc::from([0x42; 4]), false)
        .unwrap();
    assert!(cache.buffer_downloads.is_empty());
    assert_eq!(cache.device.scheduled_tick(), 0);
    cpu_write(&mut cache, &allocation, 128, &[0x5a; 4]).unwrap();
    cache
        .report_memory(ranges(192), 0, &Arc::from([0x71; 4]), true)
        .unwrap();
    assert!(!cache.buffer_downloads.is_empty());
    let work = cache.flush().unwrap().unwrap();
    cache.device.wait_for_work(work, Duration::MAX).unwrap();
    cache.progress_readbacks().unwrap();
    let bytes = cpu_read(&mut cache, &whole);
    assert_eq!(&bytes[64..68], &[0x37; 4]);
    assert_eq!(&bytes[128..132], &[0x5a; 4]);
    assert_eq!(&bytes[192..196], &[0x71; 4]);
    assert_eq!(&bytes[68..128], &[0x19; 60]);
    assert_eq!(cache.statistics.native_readback_bytes, 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn native_scalar_read_materializes_current_image_bytes_once_for_neighbouring_cpu_accesses() {
    use nixe_memory::{
        AccessOperation, AccessResult, AddressSpace, AddressWidth, GuestVirtualAddress,
        MemoryPermissions,
    };
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(&cache, 4096, 4096, 0x41);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(
            image_geometry(backing.clone(), description, &planes),
            description,
            planes,
        )
        .unwrap();
    let (_source, source_backing) = backing_allocation(&cache, 4096, 4096, 0x87);
    copy_fixture_image(&mut cache, image, source_backing);
    cpu_write(&mut cache, &allocation, 60, &[0x5a]).unwrap();
    let AccessResult::NativeRead { bytes, area } = cache
        .access_memory(backing.pin(32, 1).unwrap(), AccessOperation::ReadNative)
        .unwrap()
    else {
        panic!("native scalar result");
    };
    assert_eq!(&*bytes, &[0x87]);
    let space = AddressSpace::new(cache.memory.clone(), AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    assert_eq!(
        space
            .read_native(
                &native,
                GuestVirtualAddress::new(PAGE_SIZE + 32),
                1,
                area.as_ref()
            )
            .unwrap(),
        Some(0x87)
    );
    for (offset, expected) in [
        (33, 0x87),
        (32, 0x87),
        (47, 0x87),
        (48, 0x41),
        (55, 0x41),
        (56, 0x87),
        (60, 0x5a),
        (72, 0x41),
        (80, 0x87),
        (95, 0x87),
        (96, 0x41),
    ] {
        assert_eq!(
            space
                .read_native(
                    &native,
                    GuestVirtualAddress::new(PAGE_SIZE + offset),
                    1,
                    None
                )
                .unwrap(),
            Some(expected),
            "offset={offset}: neighbour must be direct coherent RAM"
        );
    }
    assert_eq!(cache.statistics.native_readback_bytes, 48);
    assert_eq!(cache.statistics.canonical_readback_bytes, 47);
    drop(native);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn overlapping_buffer_growth_reuses_ranges_and_preserves_queued_native_versions() {
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, whole) = backing_allocation(&cache, 4 * PAGE_SIZE as usize, 4096, 0x19);
    let first = cache
        .buffer_for(
            whole.slice(64, 128).unwrap(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
        .unwrap();
    assert_eq!(first.offset, 64);
    let alias = cache
        .buffer_for(
            whole.slice(96, 16).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
    assert_eq!(alias.buffer, first.buffer);
    assert_eq!(alias.offset, 96);
    write(&mut cache, first.buffer, first.offset, &[0xa5; 128]);
    let old_read = cache
        .download_buffer(first.buffer, first.offset, 128)
        .unwrap();
    let last = cache
        .buffer_for(
            whole.slice(2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )
        .unwrap();
    write(&mut cache, last.buffer, last.offset + 32, &[0x42; 32]);
    let merged = cache
        .buffer_for(
            whole.slice(PAGE_SIZE / 2, 2 * PAGE_SIZE).unwrap(),
            vk::BufferUsageFlags::INDIRECT_BUFFER,
        )
        .unwrap();
    assert!(cache.buffer(first.buffer).is_err());
    assert!(cache.buffer(last.buffer).is_err());
    assert_eq!(cache.statistics().buffer_merges, 2);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    cache.flush().unwrap();
    assert_eq!(cache.device.readback(old_read).unwrap(), [0xa5; 128]);
    assert_eq!(
        cache.read_buffer(merged.buffer, 64, 128).unwrap(),
        [0xa5; 128]
    );
    assert_eq!(
        cache
            .read_buffer(merged.buffer, 2 * PAGE_SIZE + 32, 32)
            .unwrap(),
        [0x42; 32]
    );
    assert_eq!(
        cache.read_buffer(merged.buffer, PAGE_SIZE, 32).unwrap(),
        [0x19; 32]
    );
    let before = cache.statistics();
    for _ in 0..20 {
        let binding = cache
            .buffer_for(
                whole.slice(64, 32).unwrap(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .unwrap();
        assert_eq!(binding.buffer, merged.buffer);
        assert_eq!(binding.offset, 64);
    }
    assert_eq!(
        cache.statistics().buffer_allocations,
        before.buffer_allocations
    );
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        before.cpu_uploaded_bytes
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn native_dma_snapshots_all_overlapping_source_regions_before_writing() {
    use nixe_gpu::native::BufferCopyRegion;
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 8192, 4096, 0x19);
    let source = cache
        .buffer_for(mapping.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    cache.fill_buffer(source.buffer, 0, 16, 0x11223344).unwrap();
    cache
        .fill_buffer(source.buffer, 16, 16, 0xaabbccdd)
        .unwrap();
    let before = cache.statistics();
    cache
        .copy_physical_buffers(
            &mapping.slice(0, 32).unwrap(),
            &mapping.slice(16, 32).unwrap(),
            &[
                BufferCopyRegion {
                    source: 0,
                    destination: 0,
                    bytes: 16,
                },
                BufferCopyRegion {
                    source: 16,
                    destination: 16,
                    bytes: 16,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        cache.statistics().canonical_readback_bytes,
        before.canonical_readback_bytes
    );
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        before.cpu_uploaded_bytes
    );
    let bytes = cache.read_buffer(source.buffer, 0, 64).unwrap();
    assert_eq!(&bytes[..16], &0x11223344u32.to_ne_bytes().repeat(4));
    assert_eq!(&bytes[16..32], &0x11223344u32.to_ne_bytes().repeat(4));
    assert_eq!(&bytes[32..48], &0xaabbccddu32.to_ne_bytes().repeat(4));
    assert_eq!(&bytes[48..], &[0x19; 16]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn four_consecutive_overlaps_expand_to_live_megabyte_coverage_and_idle_resets_growth() {
    let Some(mut cache) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_allocation, mapping) = backing_allocation(&cache, 2 * 1024 * 1024, 4096, 0x19);
    let mut range = cache
        .buffer_for_deferred_import(
            mapping.slice(0, 4096).unwrap(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
    for page in 1..=4 {
        let old = range.buffer;
        range = cache
            .buffer_for_deferred_import(
                mapping.slice(0, (page + 1) * 4096).unwrap(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .unwrap();
        assert!(!cache.buffer_binding_current(old));
        if page < 4 {
            assert_eq!(
                cache.buffer(range.buffer).unwrap().native.size(),
                (page + 1) * 4096
            );
        }
    }
    let native = cache.buffer(range.buffer).unwrap();
    assert_eq!(native.growth, 4);
    let coverage = native.contents.geometry[0].mapping();
    assert_eq!(coverage.spans()[0].start.get() % (1024 * 1024), 0);
    assert_eq!(coverage.bytes(), 1024 * 1024);
    let (_allocation2, second) = backing_allocation(&cache, 32 * 4096, 4096, 0x19);
    let mut previous = cache
        .buffer_for_deferred_import(
            second.slice(0, 4096).unwrap(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
        .unwrap();
    for page in 1..=5 {
        cache.epoch += 2;
        previous = cache
            .buffer_for_deferred_import(
                second.slice(0, (page + 1) * 4096).unwrap(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )
            .unwrap();
        assert_eq!(cache.buffer(previous.buffer).unwrap().growth, 0);
    }
    assert_eq!(
        cache.buffer(previous.buffer).unwrap().native.size(),
        6 * 4096
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn shared_budget_contraction_collects_idle_resources_but_keeps_current_effect() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_a, a) = backing_allocation(&cache, 4096, 4096, 0x11);
    let (_b, b) = backing_allocation(&cache, 4096, 4096, 0x22);
    let a_id = buffer(&mut cache, a.clone());
    cache.fill_buffer(a_id, 0, 16, 0x12345678).unwrap();
    let b_id = buffer(&mut cache, b);
    let buffers_used = cache.budget.used();
    let (_c, c) = backing_allocation(&cache, 4096, 4096, 0x33);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(image_geometry(c, description, &planes), description, planes)
        .unwrap();
    assert!(
        cache.budget.used() > buffers_used,
        "images charge the same budget"
    );
    let work = cache.flush().unwrap().unwrap();
    cache.device.wait_for_work(work, Duration::MAX).unwrap();
    cache.progress_readbacks().unwrap();
    cache.epoch += 121;
    cache.constructing.push(Resource::Buffer(b_id));
    cache.budget_ceiling = 4096;
    cache.maintain_budget(true).unwrap();
    assert!(cache.buffer(a_id).is_err());
    assert!(cache.image(image).is_err());
    assert!(cache.buffer(b_id).is_ok());
    assert_eq!(
        &cpu_read(&mut cache, &a)[..16],
        &0x12345678u32.to_ne_bytes().repeat(4)
    );
    cache.unpin(Resource::Buffer(b_id));
    cache.maintain_budget(true).unwrap();
    assert!(cache.buffer(b_id).is_err());
    assert_eq!(cache.budget.used(), 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn compatible_fragmented_buffer_overlaps_join_without_reading_gaps() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (_a, a) = backing_allocation(&cache, 4096, 4096, 0x11);
    let (_b, b) = backing_allocation(&cache, 4096, 4096, 0x22);
    let (_c, c) = backing_allocation(&cache, 4096, 4096, 0x33);
    let first =
        PhysicalMapping::join(&[a.slice(16, 16).unwrap(), b.slice(32, 16).unwrap()]).unwrap();
    let second =
        PhysicalMapping::join(&[b.slice(32, 16).unwrap(), c.slice(48, 16).unwrap()]).unwrap();
    let old = buffer(&mut cache, first);
    cache.fill_buffer(old, 16, 16, 0xaabbccdd).unwrap();
    let joined = cache
        .buffer_for(second, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    assert!(!cache.buffer_binding_current(old));
    assert_eq!(joined.offset, 16);
    assert_eq!(cache.buffer(joined.buffer).unwrap().native.size(), 48);
    let bytes = cache.read_buffer(joined.buffer, 0, 48).unwrap();
    assert_eq!(&bytes[..16], &[0x11; 16]);
    assert_eq!(&bytes[16..32], &0xaabbccddu32.to_ne_bytes().repeat(4));
    assert_eq!(&bytes[32..], &[0x33; 16]);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn retained_buffer_uploads_only_requested_dirty_pages_without_losing_other_changes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(&cache, 8 * 4096, 4096, 0x19);
    let range = cache
        .buffer_for_deferred_import(mapping.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    cache.ensure_buffer_range(range.buffer, 16, 32).unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 4096);
    cpu_write(&mut cache, &allocation, 6 * 4096 + 17, &[0x63; 16]).unwrap();
    cache.ensure_buffer_range(range.buffer, 16, 32).unwrap();
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        4096,
        "a dirty distant page must not be uploaded for this binding"
    );
    cpu_write(&mut cache, &allocation, 100, &[0x71; 16]).unwrap();
    cache
        .ensure_buffer_range(range.buffer, 6 * 4096, 64)
        .unwrap();
    assert_eq!(cache.statistics().cpu_uploaded_bytes, 8192);
    assert_eq!(
        cache.read_buffer(range.buffer, 6 * 4096 + 17, 16).unwrap(),
        [0x63; 16]
    );
    assert_eq!(
        cache.read_buffer(range.buffer, 100, 16).unwrap(),
        [0x71; 16]
    );
    let uploaded = cache.statistics().cpu_uploaded_bytes;
    let observers = cache.memory.tracking_statistics().observer_registrations;
    for value in 0..100 {
        cpu_write(&mut cache, &allocation, 7 * 4096 + 17, &[value; 16]).unwrap();
        cache.ensure_buffer_range(range.buffer, 16, 32).unwrap();
    }
    assert_eq!(cache.statistics().cpu_uploaded_bytes, uploaded);
    assert_eq!(uploaded, 3 * 4096);
    assert_eq!(
        cache.memory.tracking_statistics().observer_registrations,
        observers
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn retained_dma_matches_byte_model_across_cpu_overwrites_merges_and_fences() {
    use nixe_gpu::native::BufferCopyRegion;
    let Some(mut cache) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(&cache, 16 * PAGE_SIZE as usize, 4096, 0x19);
    let mut expected = vec![0x19; whole.bytes() as usize];
    let mut seed = 0x712a_48b3_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    for step in 0..192 {
        let size = (next() % 1024 + 1) * 4;
        let source = next() % ((expected.len() - size) / 4) * 4;
        let destination = next() % ((expected.len() - size) / 4) * 4;
        match step % 3 {
            0 => {
                let value = next() as u8;
                cpu_write(&mut cache, &allocation, destination, &vec![value; size]).unwrap();
                expected[destination..destination + size].fill(value);
            }
            1 => {
                let binding = cache
                    .buffer_for_deferred_import(
                        whole.slice(destination as u64, size as u64).unwrap(),
                        vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .unwrap();
                let value = next() as u32;
                cache
                    .fill_buffer(binding.buffer, binding.offset, binding.size, value)
                    .unwrap();
                expected[destination..destination + size]
                    .copy_from_slice(&value.to_ne_bytes().repeat(size / 4));
            }
            _ => {
                cache
                    .copy_physical_buffers(
                        &whole.slice(source as u64, size as u64).unwrap(),
                        &whole.slice(destination as u64, size as u64).unwrap(),
                        &[BufferCopyRegion {
                            source: 0,
                            destination: 0,
                            bytes: size as u64,
                        }],
                    )
                    .unwrap();
                expected.copy_within(source..source + size, destination);
            }
        }
        if step % 7 == 6 {
            cache.commit_buffer_downloads().unwrap();
            cache.flush().unwrap();
            cache.drain_buffer_downloads().unwrap();
            let actual = cpu_read(&mut cache, &whole);
            let mismatch = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (a, b))| a != b);
            assert!(
                mismatch.is_none(),
                "step {step}: first mismatch {mismatch:?}"
            );
        }
    }
    assert_eq!(cpu_read(&mut cache, &whole), expected);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn gpu_only_destination_tracks_cpu_changes_after_fence_publication() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, whole) = backing_allocation(&cache, PAGE_SIZE as usize, 4096, 0x19);
    let binding = cache
        .buffer_for_deferred_import(whole.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    for value in [0x42, 0x73, 0xa5] {
        cache
            .fill_buffer(binding.buffer, binding.offset, binding.size, 0x11223344)
            .unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        cache.drain_buffer_downloads().unwrap();
        cpu_write(&mut cache, &allocation, 64, &[value; 32]).unwrap();
        // Refresh only on alternate iterations: a later GPU overwrite must
        // rearm tracking even when the old CPU observation is already invalid.
        if value != 0x73 {
            assert_eq!(
                cache.read_buffer(binding.buffer, 64, 32).unwrap(),
                [value; 32]
            );
        }
    }
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn sparse_dma_imports_only_consumed_source_pages() {
    let Some(mut cache) = cache(ApiPreference::Auto, 128 * 1024 * 1024) else {
        panic!("Vulkan unavailable")
    };
    let validation = cache.device.validation_monitor();
    let (_a, source) = backing_allocation(&cache, 1024 * 1024, 4096, 0x19);
    let (_b, destination) = backing_allocation(&cache, 4096, 4096, 0x71);
    let before = cache.statistics();
    cache
        .copy_physical_buffers(
            &source,
            &destination.slice(0, 128).unwrap(),
            &[
                nixe_gpu::native::BufferCopyRegion {
                    source: 0,
                    destination: 0,
                    bytes: 64,
                },
                nixe_gpu::native::BufferCopyRegion {
                    source: 1024 * 1024 - 64,
                    destination: 64,
                    bytes: 64,
                },
            ],
        )
        .unwrap();
    let after = cache.statistics();
    assert_eq!(
        after.cpu_uploaded_bytes - before.cpu_uploaded_bytes,
        2 * PAGE_SIZE
    );
    assert_eq!(after.gpu_copied_bytes - before.gpu_copied_bytes, 128);
    assert_eq!(
        cpu_read(&mut cache, &destination.slice(0, 128).unwrap()),
        [0x19; 128]
    );
    // Equal envelopes do not imply overlapping byte regions. Keep disjoint
    // copies native without allocating/copying an envelope-sized scratch.
    cpu_write(&mut cache, &_a, 0, &[0x31; 64]).unwrap();
    cpu_write(&mut cache, &_a, 1024 * 1024 - 64, &[0x42; 64]).unwrap();
    let before = cache.statistics();
    cache
        .copy_physical_buffers(
            &source,
            &source,
            &[
                nixe_gpu::native::BufferCopyRegion {
                    source: 0,
                    destination: 64,
                    bytes: 64,
                },
                nixe_gpu::native::BufferCopyRegion {
                    source: 1024 * 1024 - 64,
                    destination: 1024 * 1024 - 128,
                    bytes: 64,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        cache.statistics().gpu_copied_bytes - before.gpu_copied_bytes,
        128
    );
    assert_eq!(
        cpu_read(&mut cache, &source.slice(64, 64).unwrap()),
        [0x31; 64]
    );
    assert_eq!(
        cpu_read(&mut cache, &source.slice(1024 * 1024 - 128, 64).unwrap()),
        [0x42; 64]
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn bounded_lru_reaches_idle_tail_after_pinned_prefix() {
    let Some(mut cache) = cache(ApiPreference::Auto, 128 * 1024 * 1024) else {
        panic!("Vulkan unavailable")
    };
    let validation = cache.device.validation_monitor();
    let mut allocations = Vec::new();
    let mut leases = Vec::new();
    for i in 0..65 {
        let (allocation, mapping) = backing_allocation(&cache, 4096, 4096, 0x19);
        let id = cache
            .buffer_for_deferred_import(mapping, vk::BufferUsageFlags::VERTEX_BUFFER)
            .unwrap()
            .buffer;
        if i < 64 {
            leases.push(cache.buffer_lease(id).unwrap());
        }
        allocations.push(allocation);
    }
    cache.epoch += 200;
    cache.budget_ceiling = cache.allocated_bytes() - 1;
    let before = cache.statistics().evictions;
    for _ in 0..10 {
        cache.maintain_budget(true).unwrap();
    }
    let collected = cache.statistics().evictions - before;
    assert_eq!(collected, 1);
    assert!(!cache.evict_one().unwrap());
    drop(leases);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn fragmented_uploads_capture_one_transfer_without_reordering_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();

    let backings: Vec<_> = (0..17)
        .map(|byte| backing_allocation(&cache, 4096, 4096, byte))
        .collect();
    let segments: Vec<_> = backings
        .iter()
        .enumerate()
        .map(|(index, (_, mapping))| mapping.slice(11, 97 + index as u64 * 11).unwrap())
        .collect();
    let mapping = PhysicalMapping::join(&segments).unwrap();
    let expected: Vec<_> = segments
        .iter()
        .enumerate()
        .flat_map(|(index, mapping)| std::iter::repeat_n(index as u8, mapping.bytes() as usize))
        .collect();
    let id = buffer(&mut cache, mapping);
    assert_eq!(cache.statistics().cpu_uploaded_bytes, expected.len() as u64);
    assert_eq!(
        cache.read_buffer(id, 0, expected.len() as u64).unwrap(),
        expected
    );
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn owner_inline_overwrites_keep_gpu_neighbors_without_payload_retention_or_readback() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, backing) = backing_allocation(&cache, 64, 4096, 0x19);
    let id = buffer(&mut cache, backing);
    write(&mut cache, id, 0, &[0x31; 64]);
    let memory = cache.memory().clone();
    let field = AllocationRange::new(allocation.pin(), 12, 8).unwrap();
    let extents: Vec<_> = field
        .physical_extents()
        .map(|extent| (allocation.identity(), extent))
        .collect();
    let before = cache.statistics().native_readback_bytes;
    assert!(
        unsafe { memory.write_mapped_owner(&extents, nixe_memory::WriteSource::Bytes(&[0x72; 8])) }
            .unwrap()
    );
    unsafe { cache.owner_memory_written(&extents) }.unwrap();
    assert_eq!(cache.statistics().native_readback_bytes, before);
    let actual = cache.read_buffer(id, 0, 64).unwrap();
    let mut expected = [0x31; 64];
    expected[12..20].fill(0x72);
    assert_eq!(actual, expected);
    write(&mut cache, id, 0, &[0x43; 64]);
    let full = AllocationRange::new(allocation.pin(), 0, 64).unwrap();
    let extents: Vec<_> = full
        .physical_extents()
        .map(|extent| (allocation.identity(), extent))
        .collect();
    assert!(
        unsafe {
            memory.write_mapped_owner(&extents, nixe_memory::WriteSource::Bytes(&[0x54; 64]))
        }
        .unwrap()
    );
    unsafe { cache.owner_memory_written(&extents) }.unwrap();
    assert!(
        memory.try_ram_pointer(&full).unwrap().is_some(),
        "full overwrite retires obsolete device protection"
    );
    assert_eq!(cache.read_buffer(id, 0, 64).unwrap(), [0x54; 64]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn uploads_preserve_reads_recorded_before_a_cpu_overwrite() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(&cache, 8192, 4096, 0x11);
    let id = buffer(&mut cache, Arc::clone(&mapping));
    let first = cache.download_buffer(id, 0, 64).unwrap();
    cpu_write(&mut cache, &allocation, 0, &[0x22; 64]).unwrap();
    cache.ensure_buffer_range(id, 0, 64).unwrap();
    let second = cache.download_buffer(id, 0, 64).unwrap();
    cache.flush().unwrap();
    assert_eq!(cache.device.readback(first).unwrap(), [0x11; 64]);
    assert_eq!(cache.device.readback(second).unwrap(), [0x22; 64]);
    drop(cache);
    validation.check().unwrap();
}

#[test]
fn overlapping_uploads_before_the_first_use_keep_the_last_bytes() {
    let Some(mut cache) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(&cache, 8192, 4096, 0x11);
    let id = buffer(&mut cache, mapping);
    cpu_write(&mut cache, &allocation, 13, &[0x22; 37]).unwrap();
    cache.ensure_buffer_range(id, 0, 64).unwrap();
    let read = cache.download_buffer(id, 0, 64).unwrap();
    cache.flush().unwrap();
    let mut expected = [0x11; 64];
    expected[13..50].fill(0x22);
    assert_eq!(cache.device.readback(read).unwrap(), expected);
    drop(cache);
    validation.check().unwrap();
}
