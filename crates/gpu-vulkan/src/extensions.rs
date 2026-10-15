//! Narrow bindings absent from ash 0.38; no dependency on wgpu's Vulkan ABI.
//! Field order/sType verified against Vulkan-Headers v1.4.329. Replace with ash
//! declarations when its next released bindings include these extensions.
//! https://github.com/KhronosGroup/Vulkan-Headers/blob/v1.4.329/registry/vk.xml

use ash::vk;
use std::{ffi::c_void, marker::PhantomData};

pub(crate) const UNIFIED_LAYOUTS: &std::ffi::CStr = c"VK_KHR_unified_image_layouts";
pub(crate) const SHADER_FMA: &std::ffi::CStr = c"VK_KHR_shader_fma";
pub(crate) const SURFACE_MAINTENANCE_KHR: &std::ffi::CStr = c"VK_KHR_surface_maintenance1";
pub(crate) const SWAPCHAIN_MAINTENANCE_KHR: &std::ffi::CStr = c"VK_KHR_swapchain_maintenance1";

macro_rules! feature_struct {
    ($name:ident, $s_type:expr, $($field:ident),+) => {
        #[repr(C)]
        #[derive(Debug)]
        pub(crate) struct $name<'a> {
            pub s_type: vk::StructureType,
            pub p_next: *mut c_void,
            $(pub $field: vk::Bool32,)+
            _marker: PhantomData<&'a mut ()>,
        }
        impl Default for $name<'_> {
            fn default() -> Self {
                Self { s_type: vk::StructureType::from_raw($s_type), p_next: std::ptr::null_mut(),
                    $($field: vk::FALSE,)+ _marker: PhantomData }
            }
        }
        // SAFETY: repr(C) ABI and both extension chains match the pinned registry.
        unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for $name<'_> {}
        unsafe impl vk::ExtendsDeviceCreateInfo for $name<'_> {}
    }
}
feature_struct!(UnifiedLayouts, 1_000_527_000, layouts, video);
feature_struct!(ShaderFma, 1_000_579_000, float16, float32, float64);
