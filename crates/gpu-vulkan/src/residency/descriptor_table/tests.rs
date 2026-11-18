use super::*;
use nixe_memory::{AllocationRange, Machine, WriteSource, device_memory::DeviceMemory};
use std::cell::Cell;

mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

#[test]
fn live_descriptor_reads_follow_publication_and_concurrent_remaps() -> Result<(), VulkanError> {
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(8192)?;
    let memory = DeviceMemory::new(machine, 32);
    let base = 0x10000;
    let _pin = memory.map(
        base,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 8192)?]),
        true,
    )?;
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    // The descriptor straddles two physically contiguous pages.
    let address = base + 4096 - 16;
    r.texture_cache.bindings.channel.image_descriptors[0].synchronize(1, Some((address, 0)));
    let samples = Cell::new(0);
    let locations = Cell::new(0);
    let read = |r: &mut Residency<'_>| {
        r.read_descriptor(
            &mut (),
            0,
            |t| &mut t.image_descriptors[0],
            |_, _| {
                samples.set(samples.get() + 1);
                DescriptorMapping::Guarded(1)
            },
            |_, device, a| {
                locations.set(locations.get() + 1);
                Ok(DescriptorSource::Contiguous {
                    device: a,
                    host: device.span(a, 32).unwrap(),
                })
            },
            |_, _, _, _| unreachable!(),
        )
        .map(|(bytes, revision)| (*bytes, revision))
    };
    memory.write_contiguous(address, WriteSource::Bytes(&[1; 32]))?;
    let (bytes, first) = read(&mut r)?;
    assert_eq!(bytes, [1; 32]);
    let before = samples.get();
    assert_eq!(read(&mut r)?, (bytes, first));
    assert_eq!(samples.get() - before, 1);

    memory.write_contiguous(address, WriteSource::Bytes(&[2; 32]))?;
    let (bytes, second) = read(&mut r)?;
    assert_eq!(bytes, [2; 32]);
    assert!(second > first);

    let range = r.device_buffer(
        address,
        32,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        false,
        false,
    )?;
    r.fill_buffer(range.buffer, range.offset, range.size, 0x03030303)?;
    let (bytes, third) = read(&mut r)?;
    assert_eq!(bytes, [3; 32]);
    assert!(third > second);
    assert_eq!(locations.get(), 1);

    // Force a remap inside the fragmented read, after its coherence call.
    // Only the retry's bytes and revision may be published to the table.
    let mut generation = 1;
    let (bytes, revision) = r.read_descriptor(
        &mut generation,
        0,
        |t| &mut t.image_descriptors[0],
        |g, device| DescriptorMapping::Concurrent {
            gpu: *g,
            device: device.generation(),
        },
        |g, device, addr| {
            Ok(if *g == 1 {
                DescriptorSource::Virtual(addr)
            } else {
                DescriptorSource::Contiguous {
                    device: base,
                    host: device.span(base, 32).unwrap(),
                }
            })
        },
        |r, g, addr, bytes| {
            r.flush_read_span(addr, 32)?;
            bytes.fill(99);
            *g = 2;
            Ok(())
        },
    )?;
    assert_eq!(*bytes, [0; 32]);
    assert_eq!(revision, third + 1);
    drop(r);
    drop(cache);
    validation.check()?;
    Ok(())
}
