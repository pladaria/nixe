//! Saturate production residency/recording without guest presentation pacing.
use ash::vk;
use nixe_gpu::physical::PhysicalMapping;
use nixe_gpu_vulkan::{DeviceOptions, VulkanDevice, VulkanResidency};
use nixe_memory::{AllocationRange, Machine, PAGE_SIZE};
use std::time::Instant;

fn batch(cache: &mut VulkanResidency, buffer: nixe_gpu_vulkan::BufferId, commands: u32) {
    for value in 0..commands {
        cache.fill_buffer(buffer, 0, PAGE_SIZE, value).unwrap();
    }
    cache.flush().unwrap();
    assert_eq!(
        cache.read_buffer(buffer, 0, 4).unwrap(),
        (commands - 1).to_le_bytes()
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let batches: u32 = args.get(1).map_or(128, |s| s.parse().unwrap());
    let commands: u32 = args.get(2).map_or(256, |s| s.parse().unwrap());
    assert!(batches > 0 && commands > 0);
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let device = VulkanDevice::new(DeviceOptions::default(), None).unwrap();
    assert!(matches!(
        device.profile().device_type,
        vk::PhysicalDeviceType::DISCRETE_GPU
            | vk::PhysicalDeviceType::INTEGRATED_GPU
            | vk::PhysicalDeviceType::VIRTUAL_GPU
    ));
    let mut cache = VulkanResidency::new(machine, device, 16 * 1024 * 1024).unwrap();
    let mapping = PhysicalMapping::new(&[range]).unwrap();
    let buffer = cache
        .buffer_for_deferred_import(mapping, vk::BufferUsageFlags::TRANSFER_DST)
        .unwrap()
        .buffer;
    for _ in 0..4 {
        batch(&mut cache, buffer, commands);
    }
    let started = Instant::now();
    for _ in 0..batches {
        batch(&mut cache, buffer, commands);
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "{{\"batches\":{batches},\"fills_per_batch\":{commands},\"seconds\":{seconds},\"fills_per_second\":{}}}",
        f64::from(batches) * f64::from(commands) / seconds
    );
}
