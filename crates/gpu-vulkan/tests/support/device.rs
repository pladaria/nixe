use ash::vk;
use nixe_gpu_vulkan::{ApiPreference, DeviceOptions, ValidationMode, VulkanDevice};
use std::ffi::CStr;

struct ProbeInstance(ash::Instance);
impl Drop for ProbeInstance {
    fn drop(&mut self) {
        // SAFETY: discovery creates no devices, surfaces or child objects.
        unsafe { self.0.destroy_instance(None) };
    }
}

fn hardware(kind: vk::PhysicalDeviceType) -> bool {
    matches!(
        kind,
        vk::PhysicalDeviceType::DISCRETE_GPU
            | vk::PhysicalDeviceType::INTEGRATED_GPU
            | vk::PhysicalDeviceType::VIRTUAL_GPU
    )
}

pub fn device_for_test(api: ApiPreference) -> Option<VulkanDevice> {
    device_for_test_with_validation(api, ValidationMode::Enabled)
}

pub fn device_for_test_with_validation(
    api: ApiPreference,
    validation: ValidationMode,
) -> Option<VulkanDevice> {
    // Test-only discovery, before requesting validation: a host without a GPU
    // skips even if it also has no SDK. Production startup keeps strict errors.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(entry) => entry,
        Err(ash::LoadingError::LibraryLoadFailure(error)) => {
            eprintln!("SKIP Vulkan GPU test: no Vulkan runtime ({error})");
            return None;
        }
        Err(error) => panic!("Vulkan loader is broken: {error}"),
    };
    let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
        .expect("enumerate Vulkan instance extensions");
    let portability = extensions.iter().any(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) } == ash::khr::portability_enumeration::NAME);
    let names = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let mut info = vk::InstanceCreateInfo::default();
    if portability {
        info = info
            .enabled_extension_names(&names)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    // SAFETY: entry and create-info data outlive discovery; no validation or
    // device features are needed to identify hardware adapters.
    let instance = match unsafe { entry.create_instance(&info, None) } {
        Ok(instance) => ProbeInstance(instance),
        Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER) => {
            eprintln!("SKIP Vulkan GPU test: no usable Vulkan driver");
            return None;
        }
        Err(error) => panic!("Vulkan hardware discovery failed: {error:?}"),
    };
    let physical = unsafe { instance.0.enumerate_physical_devices() }
        .expect("enumerate Vulkan physical devices");
    let adapters: Vec<_> = physical
        .into_iter()
        .filter_map(|physical| {
            let properties = unsafe { instance.0.get_physical_device_properties(physical) };
            hardware(properties.device_type).then(|| {
                unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
        })
        .collect();
    if adapters.is_empty() {
        eprintln!("SKIP Vulkan GPU test: no hardware GPU available");
        return None;
    }
    let filter = std::env::var("NIXE_VULKAN_TEST_ADAPTER").ok();
    let adapter = adapters
        .into_iter()
        .find(|name| {
            filter
                .as_ref()
                .is_none_or(|filter| name.to_lowercase().contains(&filter.to_lowercase()))
        })
        .expect("NIXE_VULKAN_TEST_ADAPTER must match an available hardware GPU");
    drop(instance);
    let device = VulkanDevice::new(
        DeviceOptions {
            validation,
            api,
            adapter: Some(adapter),
        },
        None,
    )
    .expect("available hardware GPU must initialize with the required profile and validation");
    assert!(
        hardware(device.profile().device_type),
        "GPU tests must never fall back to a software device"
    );
    eprintln!(
        "Vulkan GPU test: {} ({:?})",
        device.profile().name,
        device.profile().api_path
    );
    Some(device)
}
