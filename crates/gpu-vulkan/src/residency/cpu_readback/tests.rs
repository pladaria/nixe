use super::*;
use nixe_memory::{Machine, device_memory::DeviceMemory};

mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

#[test]
fn clean_reads_follow_physical_writers_and_partial_publication() -> Result<(), VulkanError> {
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(4096)?;
    let memory = DeviceMemory::new(machine, 32);
    let backing = AllocationRange::new(allocation.pin(), 0, 4096)?;
    let base = 0x10000;
    let alias = 0x30000;
    let original = memory.map(base, Arc::from([backing.clone()]), true)?;
    let _alias = memory.map(alias, Arc::from([backing]), true)?;
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    let buffer = r.device_buffer(
        alias,
        4096,
        vk::BufferUsageFlags::TRANSFER_DST,
        false,
        false,
    )?;
    let reader = memory.reader();
    let read = |r: &mut Residency<'_>, address| -> Result<u32, VulkanError> {
        r.flush_read_span(address, 4)?;
        let mut word = [0; 4];
        reader.read_block_unsafe(address, &mut word)?;
        Ok(u32::from_le_bytes(word))
    };
    let neighbor = r.report_query(
        vec![AllocationRange::new(allocation.pin(), 128, 4)?],
        Arc::from(2u32.to_le_bytes()),
    )?;
    r.fill_buffer(buffer.buffer, buffer.offset, 4, 1)?;
    assert_eq!(read(&mut r, base)?, 1);
    assert_eq!(read(&mut r, base)?, 1);
    let mut untouched = [0xff; 4];
    reader.read_block_unsafe(base + 128, &mut untouched)?;
    assert_eq!(untouched, [0; 4]);
    #[cfg(feature = "performance-counters")]
    let hits = r.runtime.statistics.owner_read_hits;
    assert_eq!(read(&mut r, alias)?, 1);
    #[cfg(feature = "performance-counters")]
    assert_eq!(r.runtime.statistics.owner_read_hits, hits + 1);
    assert_eq!(read(&mut r, base + 128)?, 2);
    r.fill_buffer(buffer.buffer, buffer.offset + 256, 4, 9)?;
    assert_eq!(read(&mut r, base)?, 1);
    let mut batched = [0; 4];
    reader.read_block_unsafe(base + 256, &mut batched)?;
    // Reading an adjacent word submits the batch. Only a reader of the
    // produced bytes needs to wait for publication.
    assert!(
        r.buffer_cache.pending_read(alias + 256, 4).is_some() || u32::from_le_bytes(batched) == 9
    );
    assert_eq!(read(&mut r, base + 256)?, 9);
    r.fill_buffer(buffer.buffer, buffer.offset, 4, 3)?;
    assert!(r.prepare_buffer_read(alias, 4)?.is_some());
    assert_eq!(read(&mut r, base)?, 3);
    assert!(r.buffer_cache.pending_read(alias, 4).is_none());
    assert_eq!(read(&mut r, base)?, 3);
    assert!(memory.register_command_pages(base, 4));
    let fence = r.report_query(
        vec![AllocationRange::new(allocation.pin(), 0, 4)?],
        Arc::from(4u32.to_le_bytes()),
    )?;
    assert_eq!(read(&mut r, alias)?, 4);
    drop(r);
    neighbor(Ok(())).unwrap();
    fence(Ok(())).unwrap();
    let mut r = cache.access();
    let image = single_texel(&mut r, alias + 64)?;
    assert_eq!(read(&mut r, base + 64)?, 0);
    for value in [13, 27] {
        r.clear_color_image(
            image,
            0,
            vk::ClearColorValue {
                uint32: [value, 0, 0, 0],
            },
        )?;
        assert_eq!(read(&mut r, base)?, 4);
        assert_eq!(read(&mut r, base + 64)?, value);
    }
    drop(r);
    drop(original);
    let other = memory.machine().account(4 << 20)?.allocate(4096)?;
    let _remap = memory.map(
        base,
        Arc::from([AllocationRange::new(other.pin(), 0, 4096)?]),
        true,
    )?;
    let mut r = cache.access();
    assert_eq!(read(&mut r, base)?, 0);
    assert_eq!(read(&mut r, alias)?, 4);
    drop(r);
    drop(cache);
    validation.check()?;
    Ok(())
}

fn single_texel(r: &mut Residency<'_>, address: u64) -> Result<ImageId, VulkanError> {
    let extent = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };
    let plane = crate::image_layout::plane_footprint(
        r.runtime.device_memory.read_mapping(address, 4)?,
        nixe_gpu::ImageMemoryLayout::PitchLinear {
            row_pitch: 4,
            layer_stride: 4,
        },
        [0, 0],
        [1, 1],
        4,
        4,
        (0, 4),
    )?;
    r.image_for(
        vec![plane],
        NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R32_UINT,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        },
        vec![ImagePlane {
            depth_slice: 0,
            offset: 0,
            row_pitch: 4,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        }],
        None,
    )
}

#[test]
fn disjoint_publication_waits_for_the_owner_before_writing_any_plane() -> Result<(), VulkanError> {
    use nixe_memory::{
        Access, AccessRequestId, AllocationId, CapturedMemory, DeviceAccessEndpoint, MachineId,
        PAGE_SIZE,
    };
    use std::sync::mpsc;
    struct Endpoint(mpsc::Sender<DeviceAccessRequest>);
    impl DeviceAccessEndpoint for Endpoint {
        fn submit(&self, request: DeviceAccessRequest) {
            self.0.send(request).unwrap();
        }
        fn cancel(&self, _: AccessRequestId) {}
        fn allocation_pressure(&self, _: MachineId, _: u64) {}
        fn allocation_released(&self, _: MachineId, _: AllocationId) {}
        fn physical_retired(&self, _: MachineId, _: AllocationId, _: Box<[nixe_memory::Extent]>) {}
    }
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let monitor = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(2 * PAGE_SIZE)?;
    let memory = DeviceMemory::new(machine.clone(), 32);
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE)?;
    let base = 0x10000;
    let _mapping = memory.map(base, Arc::from([range.clone()]), true)?;
    let capture = CapturedMemory::from(range);
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    let buf = r.device_buffer(
        base + PAGE_SIZE,
        8,
        vk::BufferUsageFlags::TRANSFER_DST,
        false,
        false,
    )?;
    r.fill_buffer(buf.buffer, buf.offset, 8, 0x12121212)?;
    drop(r);
    let (tx, rx) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(tx));
    machine.set_device_endpoint(&endpoint);
    let src = Arc::new(*b"12345678");
    let owned = Arc::downgrade(&src);
    let access =
        machine.publish_captured(&capture, &[(4, 4), (PAGE_SIZE, 4)], move |at, dst| {
            dst.write_at(0, &src[at..at + dst.bytes()]);
        })?;
    let Access::Pending(mut pending) = access else {
        panic!("GPU-written second plane must defer")
    };
    drop(capture);
    assert!(owned.upgrade().is_some());
    assert!(pending.poll().is_none());
    let mut first = [0xff; 8];
    memory.reader().read_block_unsafe(base, &mut first)?;
    assert_eq!(first, [0; 8]);
    let mut r = cache.access();
    r.queue_device_access(rx.try_recv().unwrap())?;
    assert!(matches!(
        pending.wait().unwrap(),
        AccessResult::Published(8)
    ));
    assert!(owned.upgrade().is_none());
    memory.reader().read_block_unsafe(base, &mut first)?;
    assert_eq!(first, *b"\0\0\0\x001234");
    r.flush_read_span(base + PAGE_SIZE, 8)?;
    let mut last = [0; 8];
    memory
        .reader()
        .read_block_unsafe(base + PAGE_SIZE, &mut last)?;
    assert_eq!(last, *b"5678\x12\x12\x12\x12");
    drop(r);
    drop(cache);
    monitor.check()
}
