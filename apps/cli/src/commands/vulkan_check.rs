use nixe_gpu_vulkan::{AllocationMode, DeviceOptions, VulkanDevice};

pub struct Arguments {
    pub options: DeviceOptions,
    pub headless: bool,
}
pub fn run(arguments: Arguments) -> Result<(), String> {
    if arguments.headless {
        verify(VulkanDevice::new(arguments.options, None).map_err(|e| e.to_string())?)
    } else {
        nixe_video_winit::with_vulkan_device(arguments.options, verify)
    }
}
fn verify(mut device: VulkanDevice) -> Result<(), String> {
    log::info!(
        "Vulkan adapter: {}; API path: {:?}; queues: {:?}",
        device.profile().name,
        device.profile().api_path,
        device.queues()
    );
    log::debug!("Vulkan device profile: {:#?}", device.profile());
    if let Some(surface) = device.surface_profile() {
        log::info!(
            "Vulkan surface: {} formats, {} present modes",
            surface.formats.len(),
            surface.present_modes.len()
        );
    }
    let validation = device.validation_monitor();
    // Dedicated diagnostic only; this synchronous fixture is not a guest hot path.
    let bytes: Vec<_> = (0..1024).map(|i| (i * 37 + 13) as u8).collect();
    if device
        .transfer_round_trip(&bytes)
        .map_err(|e| e.to_string())?
        != bytes
    {
        return Err("Vulkan buffer transfer readback differs from uploaded bytes".to_owned());
    }
    for mode in [AllocationMode::Managed, AllocationMode::Dedicated] {
        if device
            .image_round_trip(16, 16, &bytes, mode)
            .map_err(|e| e.to_string())?
            != bytes
        {
            return Err(format!(
                "Vulkan {mode:?} image transfer readback differs from uploaded bytes"
            ));
        }
    }
    drop(device);
    validation.check().map_err(|e| e.to_string())?;
    log::info!("Vulkan buffer/image transfers and cleanup passed");
    Ok(())
}
