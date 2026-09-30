//! Vulkan interoperation is isolated here; wgpu owns submission and presentation.
mod device;
mod fma;

pub(crate) use device::create_device;
