use crate::{
    ValidationMode, VulkanError,
    profile::{ApiPreference, ExtensionSet},
    validation::{self, ValidationState},
};
use ash::{Entry, Instance, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{ffi::CStr, sync::Arc};

pub trait NativeWindow: HasDisplayHandle + HasWindowHandle + Send + Sync {}
impl<T: HasDisplayHandle + HasWindowHandle + Send + Sync> NativeWindow for T {}

/// Retains the native window/display for the entire Vulkan surface lifetime.
/// Type erasure is confined to startup/window ownership, outside GPU recording.
#[derive(Clone)]
pub struct SurfaceTarget(Arc<dyn NativeWindow>);
impl SurfaceTarget {
    pub fn new<W: NativeWindow + 'static>(window: Arc<W>) -> Self {
        Self(window)
    }
}

pub(crate) struct Surface {
    pub loader: ash::khr::surface::Instance,
    pub handle: vk::SurfaceKHR,
    _window: SurfaceTarget,
}

pub(crate) struct InstanceOwner {
    pub entry: Entry,
    pub instance: Instance,
    pub version: u32,
    pub surface: Option<Surface>,
    pub validation: Arc<ValidationState>,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_maintenance_khr: bool,
    pub surface_maintenance_ext: bool,
    pub debug_utils: bool,
}

impl InstanceOwner {
    pub fn new(
        validation_mode: ValidationMode,
        api: ApiPreference,
        target: Option<SurfaceTarget>,
    ) -> Result<Arc<Self>, VulkanError> {
        // SAFETY: ash retains the dynamically loaded library for all its functions.
        let entry =
            unsafe { Entry::load() }.map_err(|e| VulkanError::Loader(e.to_string().into()))?;
        // SAFETY: entry is live; these queries have no external synchronization.
        let available = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|e| VulkanError::vk("vkEnumerateInstanceVersion", e))?
            .unwrap_or(vk::API_VERSION_1_0);
        if available < vk::API_VERSION_1_2 {
            return Err(VulkanError::ApiVersion { available });
        }
        let version = if api == ApiPreference::Compatibility12 {
            vk::API_VERSION_1_2
        } else {
            available.min(vk::API_VERSION_1_3)
        };
        let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|e| VulkanError::vk("vkEnumerateInstanceExtensionProperties", e))?;
        let supported = ExtensionSet::new(&extensions);
        let mut names: Vec<&CStr> = Vec::new();
        if let Some(window) = target.as_ref() {
            let display = window
                .0
                .display_handle()
                .map_err(|e| VulkanError::WindowHandle(e.to_string().into()))?;
            let required = ash_window::enumerate_required_extensions(display.as_raw())
                .map_err(|e| VulkanError::vk("surface instance extensions", e))?;
            for &raw in required {
                // SAFETY: ash-window returns static, NUL-terminated extension names.
                let name = unsafe { CStr::from_ptr(raw) };
                if !supported.contains(name) {
                    return Err(VulkanError::MissingInstanceExtension(
                        name.to_string_lossy().into_owned(),
                    ));
                }
                names.push(name);
            }
        }
        let portability = supported.contains(ash::khr::portability_enumeration::NAME);
        if portability {
            names.push(ash::khr::portability_enumeration::NAME);
        }
        let debug_utils = supported.contains(ash::ext::debug_utils::NAME);
        if debug_utils {
            names.push(ash::ext::debug_utils::NAME);
        }
        let capabilities2 =
            target.is_some() && supported.contains(ash::khr::get_surface_capabilities2::NAME);
        if capabilities2 {
            names.push(ash::khr::get_surface_capabilities2::NAME);
        }
        let surface_maintenance_khr =
            capabilities2 && supported.contains(crate::extensions::SURFACE_MAINTENANCE_KHR);
        let surface_maintenance_ext =
            capabilities2 && supported.contains(ash::ext::surface_maintenance1::NAME);
        if surface_maintenance_khr {
            names.push(crate::extensions::SURFACE_MAINTENANCE_KHR);
        }
        if surface_maintenance_ext {
            names.push(ash::ext::surface_maintenance1::NAME);
        }
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|e| VulkanError::vk("vkEnumerateInstanceLayerProperties", e))?;
        let validation_enabled = validation_mode == ValidationMode::Enabled;
        let layer_present = layers.iter().any(
            |l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == c"VK_LAYER_KHRONOS_validation",
        );
        if validation_enabled && (!layer_present || !debug_utils) {
            return Err(VulkanError::ValidationUnavailable);
        }
        let validation = Arc::new(ValidationState::default());
        let mut debug_info = validation::messenger_info(&validation);
        let application = vk::ApplicationInfo::default()
            .application_name(c"Nixe")
            .engine_name(c"Nixe Vulkan")
            .api_version(version);
        let extension_names: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();
        let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extension_names);
        if portability {
            info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        if validation_enabled {
            info = info
                .enabled_layer_names(&layer_names)
                .push_next(&mut debug_info);
        }
        // SAFETY: all names/pNext data live through synchronous instance creation.
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateInstance", e))?;
        let mut owner = Self {
            entry,
            instance,
            version,
            surface: None,
            validation,
            messenger: None,
            surface_maintenance_khr,
            surface_maintenance_ext,
            debug_utils,
        };
        if validation_enabled {
            let loader = ash::ext::debug_utils::Instance::new(&owner.entry, &owner.instance);
            let info = validation::messenger_info(&owner.validation);
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }
                .map_err(|e| VulkanError::vk("vkCreateDebugUtilsMessengerEXT", e))?;
            owner.messenger = Some((loader, messenger));
        }
        if let Some(window) = target {
            let display = window
                .0
                .display_handle()
                .map_err(|e| VulkanError::WindowHandle(e.to_string().into()))?;
            let handle = window
                .0
                .window_handle()
                .map_err(|e| VulkanError::WindowHandle(e.to_string().into()))?;
            // SAFETY: the retained window owns these handles, and outlives Surface.
            let surface = unsafe {
                ash_window::create_surface(
                    &owner.entry,
                    &owner.instance,
                    display.as_raw(),
                    handle.as_raw(),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("create Vulkan window surface", e))?;
            owner.surface = Some(Surface {
                loader: ash::khr::surface::Instance::new(&owner.entry, &owner.instance),
                handle: surface,
                _window: window,
            });
        }
        Ok(Arc::new(owner))
    }
}
impl Drop for InstanceOwner {
    fn drop(&mut self) {
        // SAFETY: device owners retain this instance; their devices/resources are
        // destroyed before the final Arc reaches here. Retain the window until
        // after surface destruction, and callback user data until after instance.
        unsafe {
            if let Some(surface) = self.surface.take() {
                surface.loader.destroy_surface(surface.handle, None);
            }
            if let Some((loader, messenger)) = self.messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
