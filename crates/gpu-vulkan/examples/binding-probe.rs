use nixe_gpu_vulkan::{DeviceOptions, VulkanDevice, VulkanResidency};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = VulkanDevice::new(
        DeviceOptions {
            adapter: std::env::var("NIXE_VULKAN_TEST_ADAPTER").ok(),
            ..Default::default()
        },
        None,
    )?;
    if !matches!(
        device.profile().device_type,
        ash::vk::PhysicalDeviceType::DISCRETE_GPU
            | ash::vk::PhysicalDeviceType::INTEGRATED_GPU
            | ash::vk::PhysicalDeviceType::VIRTUAL_GPU
    ) {
        return Err("binding probe requires a hardware GPU".into());
    }
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::switch_ram()?,
        device,
        64 * 1024 * 1024,
    )?;
    println!(
        "adapter={}; driver={}; validation=disabled; words=1024; dispatches=256; trials=7",
        cache.device().profile().name,
        cache.device().profile().driver_version
    );
    for result in cache.measure_bindings(256, 7)? {
        println!(
            "binding={:?}; cpu_ns_per_dispatch={:.1}; gpu_ns_per_dispatch={:.1}",
            result.binding, result.cpu_ns_per_dispatch, result.gpu_ns_per_dispatch
        );
    }
    println!("barriers={:?}", cache.barrier_statistics());
    Ok(())
}
