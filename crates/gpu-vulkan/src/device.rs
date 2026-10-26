use crate::{
    ApiPreference, DeviceProfile, SurfaceTarget, SwapchainMaintenance, ValidationMode,
    ValidationMonitor, VulkanError,
    dispatch::Dispatch,
    instance::InstanceOwner,
    profile::{self, ExtensionSet, FeatureSet},
    queue::QueueState,
};
use ash::{vk, vk::Handle};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::{
    ffi::{CStr, CString},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
    pub validation: ValidationMode,
    pub api: ApiPreference,
    /// Case-insensitive adapter-name substring. No vendor-specific selection.
    pub adapter: Option<String>,
}

#[derive(Debug)]
pub struct QueueSelection {
    pub graphics_compute_family: u32,
    pub present_family: Option<u32>,
}
#[derive(Debug)]
pub struct SurfaceProfile {
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

/// Sole queue owner. Buffers/images retain its device lifetime, not its queue.
/// Construct before GPU workers start, then move this concrete value to the owner.
pub struct VulkanDevice {
    pub(crate) owner: Arc<DeviceOwner>,
    pub(crate) submission: QueueState,
    pub(crate) dispatch: Dispatch,
    pub(crate) queue: vk::Queue,
    profile: DeviceProfile,
    queues: QueueSelection,
    surface: Option<SurfaceProfile>,
}

pub(crate) struct DeviceOwner {
    pub instance: Arc<InstanceOwner>,
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    pub queue_mutex: Arc<Mutex<()>>,
    pub allocator: Mutex<Option<Allocator>>,
    pub debug: Option<ash::ext::debug_utils::Device>,
    pub non_coherent_atom_size: u64,
    pub local_allocation_bytes: std::sync::atomic::AtomicU64,
}
impl DeviceOwner {
    pub fn name<T: Handle>(&self, handle: T, name: &str) -> Result<(), VulkanError> {
        if let Some(debug) = &self.debug {
            let name = CString::new(name)
                .map_err(|_| VulkanError::InvalidResource("debug name contains NUL"))?;
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(&name);
            // SAFETY: caller passes a live object on this device; label lives
            // through the synchronous naming call.
            unsafe { debug.set_debug_utils_object_name(&info) }
                .map_err(|e| VulkanError::vk("vkSetDebugUtilsObjectNameEXT", e))?;
        }
        Ok(())
    }
}
impl Drop for DeviceOwner {
    fn drop(&mut self) {
        // All resource wrappers retain this owner. Submitted diagnostic work
        // retains its resources until completion. The allocator must die before
        // VkDevice; no ordinary shutdown requires a device-wide idle wait here.
        self.allocator
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // SAFETY: no remaining resource/submit guard can outlive the final Arc.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

struct Candidate {
    physical: vk::PhysicalDevice,
    features: FeatureSet,
    profile: DeviceProfile,
    queues: QueueSelection,
    surface: Option<SurfaceProfile>,
    score: u32,
}

impl VulkanDevice {
    pub(crate) fn local_memory_budget(&self) -> (u64, u64) {
        let available = self
            .profile
            .enabled_extensions
            .iter()
            .any(|name| name == "VK_EXT_memory_budget");
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default();
        if available {
            properties = properties.push_next(&mut budget);
        }
        unsafe {
            self.owner
                .instance
                .instance
                .get_physical_device_memory_properties2(self.owner.physical, &mut properties);
        }
        let heaps = &properties.memory_properties.memory_heaps
            [..properties.memory_properties.memory_heap_count as usize];
        let mut capacity = 0;
        let mut usage = 0;
        for (index, heap) in heaps.iter().enumerate() {
            if !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                continue;
            }
            capacity += if available {
                budget.heap_budget[index]
            } else {
                heap.size
            };
            if available {
                usage += budget.heap_usage[index];
            }
        }
        // Driver usage includes allocator blocks, imported storage and internal
        // native objects. Allocation accounting is the fallback without EXT.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/vulkan_common/vulkan_device.cpp
        usage = usage.max(
            self.owner
                .local_allocation_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        (capacity, usage)
    }

    pub fn new(options: DeviceOptions, target: Option<SurfaceTarget>) -> Result<Self, VulkanError> {
        let instance = InstanceOwner::new(options.validation, options.api, target)?;
        // SAFETY: instance stays alive through physical device selection.
        let physical_devices = unsafe { instance.instance.enumerate_physical_devices() }
            .map_err(|e| VulkanError::vk("vkEnumeratePhysicalDevices", e))?;
        let mut failures = Vec::new();
        let mut selected: Option<Candidate> = None;
        for physical in physical_devices {
            let properties = unsafe { instance.instance.get_physical_device_properties(physical) };
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
            if let Some(filter) = &options.adapter
                && !name.to_lowercase().contains(&filter.to_lowercase())
            {
                continue;
            }
            match Self::candidate(&instance, physical, properties) {
                Ok(candidate)
                    if selected
                        .as_ref()
                        .is_none_or(|old| candidate.score > old.score) =>
                {
                    selected = Some(candidate)
                }
                Ok(_) => {}
                Err(reason) => failures.push(format!("{name}: {reason}")),
            }
        }
        let Some(mut selected) = selected else {
            if failures.is_empty() {
                failures.push(format!("no adapters match {:?}", options.adapter));
            }
            return Err(VulkanError::NoSuitableDevice(failures));
        };
        // One queue per selected family; presentation may share graphics.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/vulkan_common/vulkan_device.cpp
        let priority = [1_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(selected.queues.graphics_compute_family)
                .queue_priorities(&priority),
        ];
        if let Some(present) = selected.queues.present_family
            && present != selected.queues.graphics_compute_family
        {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(present)
                    .queue_priorities(&priority[..1]),
            );
        }
        let names: Vec<_> = selected.features.names.iter().map(|n| n.as_ptr()).collect();
        let info = selected
            .features
            .enable_info()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&names);
        // SAFETY: only supported feature bits and dependency-complete extensions
        // are enabled; selected families have nonzero queue counts.
        let device = unsafe {
            instance
                .instance
                .create_device(selected.physical, &info, None)
        }
        .map_err(|e| VulkanError::vk("vkCreateDevice", e))?;
        let debug = instance
            .debug_utils
            .then(|| ash::ext::debug_utils::Device::new(&instance.instance, &device));
        let owner = Arc::new(DeviceOwner {
            instance,
            device,
            physical: selected.physical,
            queue_mutex: Arc::new(Mutex::new(())),
            allocator: Mutex::new(None),
            debug,
            non_coherent_atom_size: selected.profile.limits.non_coherent_atom_size,
            local_allocation_bytes: std::sync::atomic::AtomicU64::new(0),
        });
        // The owner exists before any subsequent fallible creation, so failures
        // destroy the independent device and instance without a resource rollback layer.
        let dispatch = Dispatch::new(
            &owner.instance.instance,
            &owner.device,
            selected.profile.api_path,
            selected.profile.swapchain_maintenance,
            selected.profile.dynamic_state.base,
            selected.profile.attachment_feedback_loop,
            Arc::clone(&owner.queue_mutex),
        )?;
        debug_assert_eq!(
            dispatch.release_swapchain_images.is_some(),
            selected.profile.swapchain_maintenance != SwapchainMaintenance::Disabled
        );
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: owner.instance.instance.clone(),
            device: owner.device.clone(),
            physical_device: selected.physical,
            debug_settings: Default::default(),
            buffer_device_address: selected.profile.buffer_device_address,
            // Start with small slabs rather than reserving 256 MiB for the first
            // cache allocation; grow under real resource pressure.
            allocation_sizes: gpu_allocator::AllocationSizes::new(8 * 1024 * 1024, 8 * 1024 * 1024)
                .with_max_device_memblock_size(64 * 1024 * 1024)
                .with_max_host_memblock_size(16 * 1024 * 1024),
        })?;
        *owner
            .allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(allocator);
        let queue = unsafe {
            owner
                .device
                .get_device_queue(selected.queues.graphics_compute_family, 0)
        };
        owner.name(queue, "nixe graphics/compute queue")?;
        if let Some(family) = selected.queues.present_family {
            let queue = unsafe { owner.device.get_device_queue(family, 0) };
            owner.name(queue, "nixe presentation queue")?;
        }
        let submission = QueueState::new(
            Arc::clone(&owner),
            selected.queues.graphics_compute_family,
            &dispatch,
        )?;
        Ok(Self {
            owner,
            dispatch,
            queue,
            submission,
            profile: selected.profile,
            queues: selected.queues,
            surface: selected.surface,
        })
    }

    fn candidate(
        instance: &InstanceOwner,
        physical: vk::PhysicalDevice,
        properties: vk::PhysicalDeviceProperties,
    ) -> Result<Candidate, String> {
        let version = properties.api_version.min(instance.version);
        if version < vk::API_VERSION_1_2 {
            return Err("device requires Vulkan 1.2 or newer".into());
        }
        let extensions = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(physical)
        }
        .map_err(|e| format!("vkEnumerateDeviceExtensionProperties: {e:?}"))?;
        let ext = ExtensionSet::new(&extensions);
        let maintenance = if instance.surface_maintenance_khr
            && ext.contains(crate::extensions::SWAPCHAIN_MAINTENANCE_KHR)
        {
            SwapchainMaintenance::Khr
        } else if instance.surface_maintenance_ext
            && ext.contains(ash::ext::swapchain_maintenance1::NAME)
        {
            SwapchainMaintenance::Ext
        } else {
            SwapchainMaintenance::Disabled
        };
        // SAFETY: queries use only supported feature structures for this device.
        let mut features =
            unsafe { FeatureSet::query(&instance.instance, physical, &ext, version, maintenance) };
        let missing = profile::baseline_missing(
            version,
            features.v12.timeline_semaphore != 0,
            features.sync.synchronization2 != 0,
            features.rendering.dynamic_rendering != 0,
            &ext,
        );
        if !missing.is_empty() {
            return Err(format!("missing {}", missing.join(", ")));
        }
        let families = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(physical)
        };
        let mut present_support = Vec::new();
        let surface = if let Some(surface) = &instance.surface {
            if !ext.contains(ash::khr::swapchain::NAME) {
                return Err("missing VK_KHR_swapchain".into());
            }
            for i in 0..families.len() {
                present_support.push(
                    unsafe {
                        surface.loader.get_physical_device_surface_support(
                            physical,
                            i as u32,
                            surface.handle,
                        )
                    }
                    .map_err(|e| format!("surface support query: {e:?}"))?,
                );
            }
            let formats = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_formats(physical, surface.handle)
            }
            .map_err(|e| format!("surface formats: {e:?}"))?;
            let present_modes = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_present_modes(physical, surface.handle)
            }
            .map_err(|e| format!("present modes: {e:?}"))?;
            let capabilities = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_capabilities(physical, surface.handle)
            }
            .map_err(|e| format!("surface capabilities: {e:?}"))?;
            if formats.is_empty()
                || present_modes.is_empty()
                || !capabilities
                    .supported_usage_flags
                    .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                return Err(
                    "surface has no usable color-attachment presentation configuration".into(),
                );
            }
            features.names.push(ash::khr::swapchain::NAME);
            if ext.contains(ash::khr::swapchain_mutable_format::NAME) {
                features
                    .names
                    .push(ash::khr::swapchain_mutable_format::NAME);
            }
            Some(SurfaceProfile {
                formats,
                present_modes,
                capabilities,
            })
        } else {
            None
        };
        let (graphics_compute_family, present_family) = profile::choose_queues(
            &families,
            surface.as_ref().map(|_| present_support.as_slice()),
        )
        .ok_or_else(|| {
            "no graphics+compute family and compatible presentation family".to_string()
        })?;
        features.finish_enabling(&ext, version, maintenance);
        let profile =
            unsafe { features.profile(&instance.instance, physical, version, maintenance) };
        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            _ => 1,
        };
        Ok(Candidate {
            physical,
            features,
            profile,
            queues: QueueSelection {
                graphics_compute_family,
                present_family,
            },
            surface,
            score,
        })
    }
    pub fn profile(&self) -> &DeviceProfile {
        &self.profile
    }
    pub fn queues(&self) -> &QueueSelection {
        &self.queues
    }
    pub fn surface_profile(&self) -> Option<&SurfaceProfile> {
        self.surface.as_ref()
    }
    pub fn validation_monitor(&self) -> ValidationMonitor {
        ValidationMonitor(Arc::clone(&self.owner.instance.validation))
    }
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        // SAFETY: the owning instance and selected physical identity remain live.
        unsafe {
            self.owner
                .instance
                .instance
                .get_physical_device_format_properties(self.owner.physical, format)
        }
    }
}
