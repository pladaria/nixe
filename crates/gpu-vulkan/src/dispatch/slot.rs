//! Worker-owned command pools, grown four buffers at a time and reused by tick.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_command_pool.cpp
use crate::{device::DeviceOwner, queue::progress::NativeFailure};
use ash::vk;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) struct CommandSlot {
    pool: Arc<CommandPool>,
    first: usize,
    descriptor_pools: Mutex<Vec<Arc<crate::cache::DescriptorPool>>>,
}
struct CommandPool {
    owner: Arc<DeviceOwner>,
    family: u32,
    native: OnceLock<Result<Pool, NativeFailure>>,
}
struct Pool {
    owner: Arc<DeviceOwner>,
    handle: vk::CommandPool,
    commands: [vk::CommandBuffer; 4],
}
impl CommandSlot {
    pub(super) fn retain_descriptor_pool(&self, pool: Arc<crate::cache::DescriptorPool>) {
        self.descriptor_pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(pool);
    }
    pub(crate) fn retire_descriptor_pools(&self) {
        self.descriptor_pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
    pub fn owner(&self) -> &Arc<DeviceOwner> {
        &self.pool.owner
    }
    pub fn family(&self) -> u32 {
        self.pool.family
    }
    pub fn allocate(owner: Arc<DeviceOwner>, family: u32) -> [Arc<Self>; 2] {
        let pool = Arc::new(CommandPool {
            owner,
            family,
            native: OnceLock::new(),
        });
        std::array::from_fn(|index| {
            Arc::new(Self {
                pool: Arc::clone(&pool),
                first: index * 2,
                descriptor_pools: Mutex::new(Vec::new()),
            })
        })
    }
    pub(super) fn begin(
        &self,
        device: &ash::Device,
    ) -> Result<[vk::CommandBuffer; 2], NativeFailure> {
        let pool = self
            .pool
            .native
            .get_or_init(|| {
                let owner = &self.pool.owner;
                let fail = |operation, result| NativeFailure::Vulkan { operation, result };
                let handle = unsafe {
                    owner.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(
                                vk::CommandPoolCreateFlags::TRANSIENT
                                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                            )
                            .queue_family_index(self.pool.family),
                        None,
                    )
                }
                .map_err(|e| fail("vkCreateCommandPool", e))?;
                let mut pool = Pool {
                    owner: Arc::clone(owner),
                    handle,
                    commands: [vk::CommandBuffer::null(); 4],
                };
                let commands = unsafe {
                    owner.device.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(handle)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(4),
                    )
                }
                .map_err(|e| fail("vkAllocateCommandBuffers", e))?;
                pool.commands.copy_from_slice(&commands);
                Ok(pool)
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let commands = [pool.commands[self.first], pool.commands[self.first + 1]];
        for command in commands {
            // Begin implicitly resets this completed buffer with the pool's
            // RESET_COMMAND_BUFFER flag; the other pair may still be in flight.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkBeginCommandBuffer.html
            unsafe {
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(|result| NativeFailure::Vulkan {
                operation: "vkBeginCommandBuffer",
                result,
            })?;
        }
        Ok(commands)
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        // Slots retain the shared pool through queued work and actual GPU use,
        // including worker panic and partial initialization.
        unsafe { self.owner.device.destroy_command_pool(self.handle, None) };
    }
}
