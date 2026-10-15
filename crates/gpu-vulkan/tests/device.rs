use ash::vk;
use nixe_gpu_vulkan::{AllocationMode, ApiPath, ApiPreference, MemoryLocation, VulkanDevice};

#[path = "support/device.rs"]
mod support;
use support::device_for_test;

fn exercise(api: ApiPreference) {
    let Some(mut device) = device_for_test(api) else {
        return;
    };
    let validation = device.validation_monitor();
    if api == ApiPreference::Compatibility12 {
        assert_eq!(device.profile().api_path, ApiPath::Khr12);
    }
    let bytes: Vec<_> = (0..1024).map(|i| (i * 37 + 13) as u8).collect();
    assert_eq!(device.transfer_round_trip(&bytes).unwrap(), bytes);
    for mode in [AllocationMode::Managed, AllocationMode::Dedicated] {
        assert_eq!(
            device.image_round_trip(16, 16, &bytes, mode).unwrap(),
            bytes
        );
    }
    assert!(
        device
            .allocate_buffer(
                0,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                AllocationMode::Managed,
                "invalid"
            )
            .is_err()
    );
    // Fail after allocation/binding during naming; RAII must release both.
    assert!(
        device
            .allocate_buffer(
                256,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                AllocationMode::Managed,
                "bad\0name"
            )
            .is_err()
    );
    assert_eq!(device.transfer_round_trip(&bytes).unwrap(), bytes);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn independent_core_device_transfers_and_cleans_up() {
    exercise(ApiPreference::Auto);
}
#[test]
fn independent_khr_device_transfers_and_cleans_up() {
    exercise(ApiPreference::Compatibility12);
}
#[test]
fn allocations_retain_device_and_instance_until_their_last_drop() {
    let Some(device) = device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let buffer = device
        .allocate_buffer(
            257,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            AllocationMode::Dedicated,
            "retained buffer",
        )
        .unwrap();
    let image = device
        .allocate_image_2d(
            16,
            16,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AllocationMode::Dedicated,
            "retained image",
        )
        .unwrap();
    assert!(buffer.is_dedicated() && image.is_dedicated());
    drop(device);
    drop(buffer);
    drop(image);
    validation.check().unwrap();
}
#[test]
fn new_backend_has_no_wgpu_or_guest_decoder_dependency() {
    fn movable<T: Send>() {}
    movable::<VulkanDevice>();
    movable::<nixe_gpu_vulkan::AllocatedBuffer>();
    movable::<nixe_gpu_vulkan::AllocatedImage>();
    let manifest = include_str!("../Cargo.toml");
    for dependency in ["wgpu", "nixe-gpu-wgpu", "nixe-gpu-maxwell", "nixe-horizon"] {
        assert!(!manifest.contains(dependency));
    }
}
