use super::*;
use crate::ApiPreference;
use nixe_memory::{Allocation, AllocationRange, Machine, PAGE_SIZE};
use std::time::Duration;

#[test]
fn nvmap_unmap_invalidates_registered_buffers_and_cancels_committed_publications() {
    let Some(mut owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = owner.device().validation_monitor();
    let (allocation, backing) = backing_allocation(owner.device_memory(), 4096, 0x19);
    let retained = AllocationRange::new(allocation.pin(), 0, 4096).unwrap();
    let address = allocation.address();
    let mut cache = owner.access();
    let id = buffer(&mut cache, Arc::clone(&backing));
    cache.fill_buffer(id, 8, 8, 0x31313131).unwrap();
    cache.commit_buffer_downloads().unwrap();
    cache.flush().unwrap();
    drop(cache);

    // DeviceMemoryManager::Unmap invalidates caches before removing backing.
    // BufferCache::WriteMemory cancels outstanding async publication masks.
    // The already submitted staging copy keeps its original physical input
    // retained until native completion, even when publication was cancelled.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L244-L247
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L122-L128
    drop(allocation);
    let mut cache = owner.access();
    assert!(
        cache
            .buffer_cache
            .memory_tracker
            .cpu_modified(address, 4096)
    );
    assert!(
        !cache
            .buffer_cache
            .gpu_modified_ranges
            .intersects(address, 4096)
    );
    assert!(
        cache.buffer_cache.binding_current(id),
        "WriteMemory invalidates content without deleting the registered buffer"
    );
    crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
    assert_eq!(cache.statistics().native_readback_bytes, 8);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    let bytes = cpu_read(&mut cache, &backing);
    assert!(bytes.iter().all(|&byte| byte == 0x19));
    drop(cache);
    drop(retained);
    drop(owner);
    validation.check().unwrap();
}

#[test]
fn refreshing_buffer_window_keeps_valid_native_image_bytes_without_copying_them_again() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation, backing) = backing_allocation(cache.device_memory(), 4096, 0x41);
    let (mut description, planes) = rgba_image();
    description.usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let image = cache
        .image_for(
            image_geometry(backing.clone(), description, &planes),
            description,
            planes,
            None,
        )
        .unwrap();
    let (_source, source) = backing_allocation(cache.device_memory(), 4096, 0x87);
    copy_fixture_image(&mut cache, image, source);
    let alias = buffer(&mut cache, backing.slice(0, 128).unwrap());
    let mut expected = cache.read_buffer(alias, 0, 128).unwrap();
    let clear_view = cache.attachment_view(image, 0).unwrap();
    cache
        .clear_view_rect(
            clear_view,
            0,
            vk::Rect2D {
                offset: vk::Offset2D { x: 1, y: 1 },
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [1.; 4] },
            },
        )
        .unwrap();
    let before = cache.statistics();
    cache.ensure_buffer_range(alias, 60, 4).unwrap();
    assert_eq!(
        cache.statistics().gpu_copied_bytes,
        before.gpu_copied_bytes + 4
    );
    assert_eq!(
        cache.statistics().cpu_uploaded_bytes,
        before.cpu_uploaded_bytes
    );
    expected[60..64].fill(0xff);
    assert_eq!(cache.read_buffer(alias, 0, 128).unwrap(), expected);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn cpu_streaming_buffer_imports_keep_independent_aliases_and_gpu_handoffs() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn cpu_updates_survive_disjoint_gpu_writes_on_the_same_page() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let validation = cache_owner.device().validation_monitor();
    for (offset, size, submit) in [(64, 256, false), (3840, 412, false), (3840, 412, true)] {
        let (allocation, whole) = backing_allocation(cache_owner.device_memory(), 8192, 0);
        let mut cache = cache_owner.access();
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
    let cache = cache_owner.access();
    // Fence publication can retire the three exact 32-byte writes, never
    // the unrelated bytes of their shared page.
    assert!(cache.statistics().canonical_readback_bytes <= 3 * 32);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn deferred_buffer_import_observes_writes_after_preparation() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn explicit_scattered_buffers_preserve_fragment_order_and_repeated_physical_bytes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_a, a) = backing_allocation(cache.device_memory(), 4096, 0x31);
    let (_b, b) = backing_allocation(cache.device_memory(), 4096, 0x42);
    let (_c, c) = backing_allocation(cache.device_memory(), 4096, 0x53);
    assert_eq!(
        a.spans()[0].start.get() / 65536,
        c.spans()[0].start.get() / 65536
    );
    let left = PhysicalMapping::join(&[
        a.slice(32, 16).unwrap(),
        b.slice(64, 16).unwrap(),
        a.slice(32, 16).unwrap(),
        b.slice(128, 16).unwrap(),
    ])
    .unwrap();
    let right =
        PhysicalMapping::join(&[a.slice(32, 16).unwrap(), c.slice(64, 16).unwrap()]).unwrap();
    let left_id = buffer(&mut cache, Arc::clone(&left));
    let right_id = buffer(&mut cache, Arc::clone(&right));
    assert_ne!(left_id, right_id);
    cache.fill_buffer(left_id, 16, 16, 0x11223344).unwrap();
    cache.fill_buffer(right_id, 16, 16, 0x55667788).unwrap();
    assert_eq!(
        cache.read_buffer(left_id, 16, 16).unwrap(),
        0x11223344u32.to_ne_bytes().repeat(4)
    );
    assert_eq!(
        cache.read_buffer(right_id, 16, 16).unwrap(),
        0x55667788u32.to_ne_bytes().repeat(4)
    );
    drop(cache);
    drop(cache_owner);
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

    Some(
        VulkanResidency::new(
            nixe_memory::device_memory::DeviceMemory::new(
                Machine::new(64 * 1024 * 1024).unwrap(),
                34,
            ),
            device,
            budget,
        )
        .unwrap(),
    )
}
/// A fixture NvMap allocation: CPU ownership and its real SMMU pin share scope.
/// Cache tests therefore exercise DAddr selection instead of an implicit
/// physical-buffer lookup that no longer exists.
#[derive(Clone)]
pub(crate) struct MappedAllocation {
    _device: Arc<(
        nixe_memory::device_memory::DeviceMapping,
        nixe_memory::device_memory::DeviceReservation,
    )>,
    allocation: Allocation,
}
impl MappedAllocation {
    pub(crate) fn address(&self) -> u64 {
        self._device.0.address()
    }
}
impl std::ops::Deref for MappedAllocation {
    type Target = Allocation;
    fn deref(&self) -> &Allocation {
        &self.allocation
    }
}
pub(crate) fn backing_allocation(
    device: &nixe_memory::device_memory::DeviceMemory,
    size: usize,
    byte: u8,
) -> (MappedAllocation, Arc<PhysicalMapping>) {
    let machine = device.machine();
    let bytes = (size as u64).next_multiple_of(PAGE_SIZE);
    let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, size as u64).unwrap();
    machine
        .write_resolved(std::slice::from_ref(&range), &vec![byte; size])
        .unwrap();
    let mapping = PhysicalMapping::new(&[range]).unwrap();
    let reservation = device
        .reserve(bytes.div_ceil(65536) * 65536, 65536)
        .unwrap();
    let input = AllocationRange::new(allocation.pin(), 0, bytes).unwrap();
    let mapped = device
        .map(reservation.address(), vec![input].into(), true)
        .unwrap();
    (
        MappedAllocation {
            _device: Arc::new((mapped, reservation)),
            allocation,
        },
        mapping,
    )
}
pub(crate) fn cpu_read(cache: &mut Residency<'_>, range: &PhysicalMapping) -> Vec<u8> {
    let inputs = range.pin(0, range.bytes()).unwrap();
    let nixe_memory::AccessResult::Read(bytes) = cache
        .access_memory(&inputs, &mut nixe_memory::AccessOperation::Read(Vec::new()))
        .unwrap()
    else {
        panic!("expected read")
    };
    bytes
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

pub(crate) fn footprint_read(cache: &mut Residency<'_>, footprint: Arc<Footprint>) -> Vec<u8> {
    let bytes = footprint_bytes(&footprint);
    let buffer = footprint_buffer(cache, footprint);
    cache.read_buffer(buffer, 0, bytes).unwrap()
}
pub(crate) fn cpu_write(
    cache: &mut Residency<'_>,
    allocation: &Allocation,
    offset: impl TryInto<u64>,
    bytes: &[u8],
) -> Result<(), VulkanError> {
    let offset = offset.try_into().ok().expect("fixture offset");
    let input = AllocationRange::new(allocation.pin(), offset, bytes.len() as u64)?;
    let result = cache.access_memory(
        &[input],
        &mut nixe_memory::AccessOperation::Write(bytes.into()),
    )?;
    assert!(matches!(result, nixe_memory::AccessResult::Written(_)));
    Ok(())
}
fn write(cache: &mut Residency<'_>, id: BufferId, offset: u64, bytes: &[u8]) -> WorkTicket {
    cache.write_buffer(id, offset, bytes).unwrap()
}
#[test]
fn cached_ram_reopens_on_write_and_preserves_later_device_store_collection() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let device_memory = cache_owner.device_memory().clone();
    let memory = device_memory.machine().clone();
    let allocation = memory.account(8192).unwrap().allocate(8192).unwrap();
    let space =
        nixe_memory::AddressSpace::new(memory.clone(), nixe_memory::AddressWidth::Bits32).unwrap();
    for base in [4096, 16384] {
        space
            .map(
                nixe_memory::GuestVirtualAddress::new(base),
                allocation.clone(),
                0,
                8192,
                nixe_memory::MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let inputs = space
        .capture(
            nixe_memory::GuestVirtualAddress::new(16384),
            8192,
            nixe_memory::MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    memory.write_resolved(&inputs, &[0x19; 8192]).unwrap();
    let reservation = device_memory.reserve(65536, 65536).unwrap();
    let pin = device_memory
        .map(reservation.address(), inputs.into(), true)
        .unwrap();
    let mut cache = cache_owner.access();
    let id = buffer(&mut cache, Arc::clone(pin.physical()));
    // Actual NvMap captures identify the CPU mapping protected by cached counts.
    drop(cache);
    memory.enable_cpu_write_collection().unwrap();
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    assert!(
        !space
            .write_native(
                &native,
                nixe_memory::GuestVirtualAddress::new(16384 + 4094),
                &[1, 2, 3, 4, 5, 6],
            )
            .unwrap()
    );
    drop(native);
    assert!(matches!(
        space
            .write(
                nixe_memory::GuestVirtualAddress::new(16384 + 4094),
                &[1, 2, 3, 4, 5, 6]
            )
            .unwrap(),
        nixe_memory::Access::Ready(())
    ));
    // Reopened RAM permits later direct stores, without an exact GPU journal.
    let native = core.enter_native().unwrap();
    assert!(
        space
            .write_native(
                &native,
                nixe_memory::GuestVirtualAddress::new(4096 + 4095),
                &[2]
            )
            .unwrap()
    );
    drop(native);
    memory.gather_gpu_dirty_memory(|_, _| panic!("unexpected pending CPU invalidation"));
    memory.gather_gpu_dirty_memory(|address, bytes| {
        nixe_memory::device_memory::DeviceMemoryInterface::invalidate_region(
            &*cache_owner.caches,
            address,
            bytes,
        )
    });
    let mut cache = cache_owner.access();
    assert_eq!(
        cache.read_buffer(id, 4093, 8).unwrap(),
        [0x19, 1, 2, 3, 4, 5, 6, 0x19]
    );

    // A later GPU producer replaces the RAM version. Its already committed
    // fence must preserve still later native stores into the same bytes.
    write(&mut cache, id, 4094, &[11; 6]);
    cache.commit_buffer_downloads().unwrap();
    cache.flush().unwrap();
    drop(cache);
    let native = core.enter_native().unwrap();
    let target = nixe_memory::GuestVirtualAddress::new(16384 + 4095);
    assert!(!space.write_native(&native, target, &[77, 78]).unwrap());
    let operation = space
        .capture_native(&native, target, 2, nixe_memory::MemoryPermissions::WRITE)
        .unwrap();
    let cached = space.has_cached_pages(&native, target, 2).unwrap();
    native.suspend();
    assert!(matches!(
        native
            .access_captured(
                &operation,
                cached,
                nixe_memory::AccessOperation::Write(vec![77, 78])
            )
            .unwrap()
            .into_ready()
            .unwrap(),
        nixe_memory::AccessResult::Written(_)
    ));
    native.resume().unwrap();
    drop(native);
    let mut cache = cache_owner.access();
    crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
    assert_eq!(
        cache.read_buffer(id, 4093, 8).unwrap(),
        [0x19, 11, 77, 78, 11, 11, 11, 0x19]
    );
    memory.gather_gpu_dirty_memory(|_, _| panic!("unexpected pending CPU invalidation"));
    memory.disable_cpu_write_collection();
    drop(cache);
    drop(cache_owner);
}

#[test]
fn a_batch_of_fragmented_cpu_overwrites_preserves_native_buffer_neighbors() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let device_memory = cache_owner.device_memory().clone();
    let memory = device_memory.machine().clone();
    let allocation = memory.account(4096).unwrap().allocate(4096).unwrap();
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
    let inputs = space
        .capture(
            nixe_memory::GuestVirtualAddress::new(4096),
            4096,
            nixe_memory::MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    memory.write_resolved(&inputs, &[0x19; 4096]).unwrap();
    let reservation = device_memory.reserve(65536, 65536).unwrap();
    let mapping = device_memory
        .map(reservation.address(), inputs.into(), true)
        .unwrap();
    let mut cache = cache_owner.access();
    let id = buffer(&mut cache, mapping.physical().clone());
    write(&mut cache, id, 0, &[11; 4096]);
    memory.enable_cpu_write_collection().unwrap();
    // The first CPU overwrite uses CachedWriteMemory; later native stores
    // collect exact byte ranges through the same original NvMap CPU mapping.
    cpu_write(&mut cache, &allocation, 3u64, &[77]).unwrap();
    drop(cache);
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    let mut expected = vec![11; 4096];
    expected[3] = 77;
    for offset in (20..4096).step_by(17) {
        let address = nixe_memory::GuestVirtualAddress::new(4096 + offset as u64);
        if !space.write_native(&native, address, &[77]).unwrap() {
            let operation = space
                .capture_native(&native, address, 1, nixe_memory::MemoryPermissions::WRITE)
                .unwrap();
            let cached = space.has_cached_pages(&native, address, 1).unwrap();
            native.suspend();
            assert!(matches!(
                native
                    .access_captured(
                        &operation,
                        cached,
                        nixe_memory::AccessOperation::Write(vec![77])
                    )
                    .unwrap()
                    .into_ready()
                    .unwrap(),
                nixe_memory::AccessResult::Written(_)
            ));
            native.resume().unwrap();
        }
        expected[offset] = 77;
    }
    drop(native);
    memory.gather_gpu_dirty_memory(|address, bytes| {
        nixe_memory::device_memory::DeviceMemoryInterface::invalidate_region(
            &*cache_owner.caches,
            address,
            bytes,
        )
    });
    let mut cache = cache_owner.access();
    assert_eq!(cache.read_buffer(id, 0, 4096).unwrap(), expected);
    memory.disable_cpu_write_collection();
    drop(cache);
    drop(cache_owner);
}

fn exercise_buffer_coherence(api: ApiPreference) {
    let Some(mut cache_owner) = cache(api, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, whole) = backing_allocation(cache.device_memory(), 3 * 4096, 0x19);
    let a = buffer(&mut cache, whole.clone());
    assert_eq!(cache.statistics().cpu_uploaded_bytes, whole.bytes());
    assert_eq!(
        cache.runtime.device.queue_usage().staging_capacity,
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
    cache.ensure_buffer_range(a, 0, whole.bytes()).unwrap();
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
    cache.ensure_buffer_range(a, 0, whole.bytes()).unwrap();
    assert_eq!(
        &cache.read_buffer(a, 130, 9).unwrap(),
        &[0xbb, 0xbb, 0xdd, 0xdd, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb]
    );
    drop(cache);
    drop(cache_owner);
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

pub(crate) fn rgba_image() -> (NativeImageDescription, Vec<ImagePlane>) {
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
fn native_compressed_blocks_mips_and_layers_preserve_exact_bytes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_13, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
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
            None,
        )
        .unwrap();
    let (_allocation_14, source) = backing_allocation(cache.device_memory(), 4096, 0x71);
    copy_fixture_image(&mut cache, image, source);
    cache.flush().unwrap();
    // Splitting one BC block requests exactly three bytes, realigned on GPU.
    let alias = buffer(&mut cache, backing.slice(3, 3).unwrap());
    assert_eq!(cache.read_buffer(alias, 0, 3).unwrap(), vec![0x71; 3]);
    let before = cache.runtime.device.queue_usage().submitted_tick;
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
        cache.runtime.device.queue_usage().submitted_tick - before,
        1,
        "all page producer rows share one demanded readback"
    );
    assert_eq!(cache.statistics().canonical_readback_bytes, 80);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn fragmented_image_import_batches_transfers_and_bounds_temporary_allocations() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (width, height) = (128_u32, 32_u32);
    let size = u64::from(width * height * 4);
    let (_allocation_19, backing) = backing_allocation(cache.device_memory(), size as usize, 0xa5);
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
        .image_for(
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
            None,
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn pending_producers_and_cpu_downloads_share_one_submission() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation_20, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
    let id = buffer(&mut cache, backing.clone());
    write(&mut cache, id, 64, &[0x71; 32]);
    let before = cache.runtime.device.queue_usage().submitted_tick;
    let mut expected = vec![0x19; 4096];
    expected[64..96].fill(0x71);
    assert_eq!(cpu_read(&mut cache, &backing), expected);
    assert_eq!(
        cache.runtime.device.queue_usage().submitted_tick,
        before + 1,
        "pending producer and canonical download must submit together"
    );

    write(&mut cache, id, 96, &[0x93; 16]);
    let before = cache.runtime.device.queue_usage().submitted_tick;
    assert_eq!(cache.read_buffer(id, 96, 16).unwrap(), [0x93; 16]);
    assert_eq!(
        cache.runtime.device.queue_usage().submitted_tick,
        before + 1,
        "native diagnostic reads must not pre-submit their producer"
    );

    write(&mut cache, id, 128, &[0x55; 16]);
    let before = cache.runtime.device.queue_usage().submitted_tick;
    assert_eq!(&cpu_read(&mut cache, &backing)[128..144], &[0x55; 16]);
    assert_eq!(
        cache.runtime.device.queue_usage().submitted_tick,
        before + 1,
        "CPU reads accept pending ownership before recording their download"
    );

    let (_allocation_21, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
    let (description, planes) = rgba_image();
    let plane = planes[0];
    let image = cache
        .image_for(
            crate::residency::tests::image_geometry(backing.clone(), description, &planes),
            description,
            planes,
            None,
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
    let before = cache.runtime.device.queue_usage().submitted_tick;
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
        cache.runtime.device.queue_usage().submitted_tick,
        before + 1,
        "pending clear and canonical image download must submit together"
    );
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn explicit_scattered_buffers_share_physical_producers_without_merging_storage() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_a, a) = backing_allocation(cache.device_memory(), 4096, 0x11);
    let (_b, b) = backing_allocation(cache.device_memory(), 4096, 0x22);
    let (_c, c) = backing_allocation(cache.device_memory(), 4096, 0x33);
    let first =
        PhysicalMapping::join(&[a.slice(16, 16).unwrap(), b.slice(32, 16).unwrap()]).unwrap();
    let second =
        PhysicalMapping::join(&[b.slice(32, 16).unwrap(), c.slice(48, 16).unwrap()]).unwrap();
    let old = buffer(&mut cache, first);
    cache.fill_buffer(old, 16, 16, 0xaabbccdd).unwrap();
    let joined = cache
        .buffer_for(second, vk::BufferUsageFlags::VERTEX_BUFFER)
        .unwrap();
    assert!(cache.buffer_cache.binding_current(old));
    assert_eq!(joined.offset, 0);
    assert_eq!(cache.read_buffer(old, 0, 16).unwrap(), [0x11; 16]);
    let bytes = cache.read_buffer(joined.buffer, 0, 32).unwrap();
    assert_eq!(&bytes[..16], &0xaabbccddu32.to_ne_bytes().repeat(4));
    assert_eq!(&bytes[16..], &[0x33; 16]);
    assert_eq!(cache.statistics().canonical_readback_bytes, 0);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn gpu_only_destination_tracks_cpu_changes_after_fence_publication() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, whole) = backing_allocation(cache.device_memory(), PAGE_SIZE as usize, 0x19);
    let binding = cache
        .buffer_for_deferred_import(whole.clone(), vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap();
    for value in [0x42, 0x73, 0xa5] {
        cache
            .fill_buffer(binding.buffer, binding.offset, binding.size, 0x11223344)
            .unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn fragmented_uploads_capture_one_transfer_without_reordering_bytes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();

    let backings: Vec<_> = (0..17)
        .map(|byte| backing_allocation(cache.device_memory(), 4096, byte))
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
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn uploads_preserve_reads_recorded_before_a_cpu_overwrite() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(cache.device_memory(), 8192, 0x11);
    let id = buffer(&mut cache, Arc::clone(&mapping));
    let first = cache.download_buffer(id, 0, 64).unwrap();
    cpu_write(&mut cache, &allocation, 0, &[0x22; 64]).unwrap();
    cache.ensure_buffer_range(id, 0, 64).unwrap();
    let second = cache.download_buffer(id, 0, 64).unwrap();
    cache.flush().unwrap();
    cache.runtime.device.flush().unwrap();
    cache
        .runtime
        .device
        .wait_for_work(first.work(), Duration::MAX)
        .unwrap();
    assert_eq!(first.completed().unwrap().as_ref(), [0x11; 64]);
    cache.runtime.device.flush().unwrap();
    cache
        .runtime
        .device
        .wait_for_work(second.work(), Duration::MAX)
        .unwrap();
    assert_eq!(second.completed().unwrap().as_ref(), [0x22; 64]);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn overlapping_uploads_before_the_first_use_keep_the_last_bytes() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, mapping) = backing_allocation(cache.device_memory(), 8192, 0x11);
    let id = buffer(&mut cache, mapping);
    cpu_write(&mut cache, &allocation, 13, &[0x22; 37]).unwrap();
    cache.ensure_buffer_range(id, 0, 64).unwrap();
    let read = cache.download_buffer(id, 0, 64).unwrap();
    cache.flush().unwrap();
    let mut expected = [0x11; 64];
    expected[13..50].fill(0x22);
    cache.runtime.device.flush().unwrap();
    cache
        .runtime
        .device
        .wait_for_work(read.work(), Duration::MAX)
        .unwrap();
    assert_eq!(read.completed().unwrap().as_ref(), expected);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn foreign_image_writes_preserve_unrelated_buffer_pages() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 32 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation, backing) =
        backing_allocation(cache.device_memory(), 3 * PAGE_SIZE as usize, 0x19);
    let (description, planes) = rgba_image();
    let image = cache
        .image_for(
            image_geometry(backing.clone(), description, &planes),
            description,
            planes,
            None,
        )
        .unwrap();
    let buffer = buffer(&mut cache, backing);
    for iteration in 0..3 {
        cache
            .clear_color_image(
                image,
                0,
                vk::ClearColorValue {
                    float32: [iteration as f32 / 2., 1., 0., 1.],
                },
            )
            .unwrap();
        let before = cache.statistics().cpu_uploaded_bytes;
        cache
            .ensure_buffer_range(buffer, 2 * PAGE_SIZE, 16)
            .unwrap();
        let copied = cache.statistics().cpu_uploaded_bytes - before;
        println!(
            "AUDIT FOLLOWUP disjoint_image_write iteration={iteration} unchanged_16byte_upload={copied}"
        );
        assert_eq!(copied, 0);
        cache.ensure_buffer_range(buffer, 32, 16).unwrap();
    }
    cache.flush().unwrap();
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn repeated_gpu_writes_keep_one_pending_interval_until_a_real_cpu_fence() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 8 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (_allocation, backing) =
        backing_allocation(cache.device_memory(), PAGE_SIZE as usize, 0x19);
    let id = buffer(&mut cache, Arc::clone(&backing));
    for value in 0..128_u32 {
        cache.write_buffer(id, 8, &value.to_le_bytes()).unwrap();
    }
    cache.write_buffer(id, 20, &[0x71; 4]).unwrap();
    cache.write_buffer(id, 12, &[0x37; 8]).unwrap();
    assert_eq!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .iter()
            .collect::<Vec<_>>(),
        [(_allocation.address() + 8, 16)]
    );
    cache.flush().unwrap();
    assert_eq!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .iter()
            .count(),
        1
    );
    assert_eq!(cache.statistics().native_readback_bytes, 0);
    cache.commit_buffer_downloads().unwrap();
    assert!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .is_empty()
    );
    cache.flush().unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[8..12], &127_u32.to_le_bytes());
    assert_eq!(&bytes[12..20], &[0x37; 8]);
    assert_eq!(&bytes[20..24], &[0x71; 4]);
    assert_eq!(cache.statistics().native_readback_bytes, 16);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn pending_device_ranges_survive_native_buffer_join_and_exact_cpu_replacement() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, backing) = backing_allocation(cache.device_memory(), 2 * 65536, 0x19);
    let base = allocation.address();
    let first = cache
        .device_buffer(base, 16, vk::BufferUsageFlags::STORAGE_BUFFER, true, false)
        .unwrap();
    let second = cache
        .device_buffer(
            base + 65536,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
            false,
        )
        .unwrap();
    assert_ne!(first.buffer, second.buffer);
    cache
        .fill_buffer(first.buffer, first.offset + 8, 4, 0x11223344)
        .unwrap();
    cache
        .fill_buffer(second.buffer, second.offset + 8, 4, 0xaabbccdd)
        .unwrap();
    cpu_write(&mut cache, &allocation, 8, &[0x9a, 0xab]).unwrap();
    let expected_ranges = [(base + 10, 2), (base + 65544, 4)];
    assert_eq!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .iter()
            .collect::<Vec<_>>(),
        expected_ranges
    );
    let joined = cache
        .device_buffer(
            base,
            65552,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
            false,
        )
        .unwrap();
    assert_ne!(joined.buffer, first.buffer);
    assert_ne!(joined.buffer, second.buffer);
    assert_eq!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .iter()
            .collect::<Vec<_>>(),
        expected_ranges
    );
    cache.commit_buffer_downloads().unwrap();
    assert!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .is_empty()
    );
    cache.flush().unwrap();
    crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[8..12], &[0x9a, 0xab, 0x22, 0x11]);
    assert_eq!(&bytes[65544..65548], &0xaabbccddu32.to_le_bytes());
    assert!(bytes[12..65544].iter().all(|&byte| byte == 0x19));
    assert_eq!(cache.statistics().canonical_readback_bytes, 6);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

#[test]
fn ordered_device_writes_download_the_latest_ranges_at_the_cpu_fence() {
    let Some(mut cache_owner) = cache(ApiPreference::Auto, 16 * 1024 * 1024) else {
        return;
    };
    let mut cache = cache_owner.access();
    let validation = cache.runtime.device.validation_monitor();
    let (allocation, backing) = backing_allocation(cache.device_memory(), 4096, 0x19);
    let id = buffer(&mut cache, Arc::clone(&backing));
    cache.fill_buffer(id, 8, 8, 0x11111111).unwrap();
    cache.order_device().unwrap();
    assert!(
        cache
            .buffer_cache
            .uncommitted_gpu_modified_ranges
            .is_empty()
    );
    cache.fill_buffer(id, 12, 8, 0x22222222).unwrap();
    cache.order_device().unwrap();
    cache.fill_buffer(id, 16, 8, 0x33333333).unwrap();
    cpu_write(&mut cache, &allocation, 9, &[0x73]).unwrap();
    assert_eq!(cache.buffer_cache.committed_gpu_modified_ranges.len(), 2);
    cache.commit_buffer_downloads().unwrap();
    assert!(cache.buffer_cache.committed_gpu_modified_ranges.is_empty());
    cache.flush().unwrap();
    crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
    let bytes = cpu_read(&mut cache, &backing);
    assert_eq!(&bytes[8..12], &[0x11, 0x73, 0x11, 0x11]);
    assert_eq!(&bytes[12..16], &[0x22; 4]);
    assert_eq!(&bytes[16..24], &[0x33; 8]);
    assert_eq!(&bytes[24..32], &[0x19; 8]);
    // Four committed intervals visit the same dirty tracker page. Eden's
    // ForEachDownloadRange intersects each page with all exact GPU ranges.
    assert_eq!(cache.statistics().canonical_readback_bytes, 4 * 15);
    drop(cache);
    drop(cache_owner);
    validation.check().unwrap();
}

pub(crate) fn wait_buffer_downloads(cache: &mut Residency<'_>) -> Result<(), VulkanError> {
    while let Some(download) = cache.buffer_cache.buffer_downloads.front() {
        if let Some(work) = download
            .as_ref()
            .and_then(super::buffer_download::Download::work)
        {
            cache
                .runtime
                .device
                .wait_for_work(work, std::time::Duration::MAX)?;
        }
        cache.buffer_cache.pop_async_flush()?;
    }
    Ok(())
}
