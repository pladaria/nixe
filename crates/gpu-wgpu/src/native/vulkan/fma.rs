//! Narrow ABI binding for an extension newer than ash 0.38, matching HAL's ash
//! version without a second incompatible Vulkan binding or a dependency fork.
//! Remove this declaration when HAL's ash version exposes the official binding.
//! https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFmaFeaturesKHR.html
//! https://docs.vulkan.org/refpages/latest/refpages/source/VkStructureType.html
use std::{ffi::c_void, marker::PhantomData};

use ash::vk;

pub(super) const NAME: &std::ffi::CStr = c"VK_KHR_shader_fma";

#[repr(C)]
pub(super) struct Features<'a> {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub float16: vk::Bool32,
    pub float32: vk::Bool32,
    pub float64: vk::Bool32,
    _marker: PhantomData<&'a mut ()>,
}

impl Default for Features<'_> {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_579_000),
            p_next: std::ptr::null_mut(),
            float16: vk::FALSE,
            float32: vk::FALSE,
            float64: vk::FALSE,
            _marker: PhantomData,
        }
    }
}

// SAFETY: repr(C), member types/order and sType match the linked registry ABI;
// both chains are explicitly listed as extension points for this structure.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Features<'_> {}
unsafe impl vk::ExtendsDeviceCreateInfo for Features<'_> {}
