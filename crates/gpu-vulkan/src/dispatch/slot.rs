//! Native recording context. Only the recording worker creates or uses it.
use crate::{device::DeviceOwner, queue::progress::NativeFailure};
use ash::vk;
use std::sync::{Arc, OnceLock};

pub(crate) struct CommandSlot {
    owner: Arc<DeviceOwner>,
    family: u32,
    native: OnceLock<Result<Pool, NativeFailure>>,
}
struct Pool {
    owner: Arc<DeviceOwner>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
}
impl CommandSlot {
    pub fn new(owner: Arc<DeviceOwner>, family: u32) -> Arc<Self> {
        Arc::new(Self {
            owner,
            family,
            native: OnceLock::new(),
        })
    }
    pub(super) fn begin(&self, device: &ash::Device) -> Result<vk::CommandBuffer, NativeFailure> {
        let pool = self
            .native
            .get_or_init(|| {
                let fail = |operation, result| NativeFailure::Vulkan { operation, result };
                let raw = unsafe {
                    self.owner.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(self.family),
                        None,
                    )
                }
                .map_err(|e| fail("vkCreateCommandPool", e))?;
                let mut pool = Pool {
                    owner: Arc::clone(&self.owner),
                    pool: raw,
                    command: vk::CommandBuffer::null(),
                };
                pool.command = unsafe {
                    self.owner.device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(raw)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                }
                .map_err(|e| fail("vkAllocateCommandBuffers", e))?[0];
                Ok(pool)
            })
            .as_ref()
            .map_err(Clone::clone)?;
        unsafe { device.reset_command_pool(pool.pool, vk::CommandPoolResetFlags::empty()) }
            .map_err(|result| NativeFailure::Vulkan {
                operation: "vkResetCommandPool",
                result,
            })?;
        unsafe {
            device.begin_command_buffer(
                pool.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|result| NativeFailure::Vulkan {
            operation: "vkBeginCommandBuffer",
            result,
        })?;
        Ok(pool.command)
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        // The command-slot lease survives queued recording and its last GPU
        // use, even if the worker panics. Destruction never depends on a reply.
        unsafe { self.owner.device.destroy_command_pool(self.pool, None) };
    }
}
