//! Diagnostic timestamps retire with their command slot. No GPU waits are added.
use super::*;
use nixe_gpu::metrics::{Counter, record};

pub(super) struct Queries {
    owner: Arc<DeviceOwner>,
    pool: vk::QueryPool,
    period: f64,
    mask: u64,
    pending: bool,
}
impl Queries {
    pub fn new(owner: Arc<DeviceOwner>, family: u32) -> Result<Option<Self>, VulkanError> {
        let families = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_queue_family_properties(owner.physical)
        };
        let bits = families[family as usize].timestamp_valid_bits;
        if bits == 0 {
            return Ok(None);
        }
        let properties = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_properties(owner.physical)
        };
        let pool = unsafe {
            owner.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("queue timing query pool", e))?;
        Ok(Some(Self {
            owner,
            pool,
            period: f64::from(properties.limits.timestamp_period),
            mask: u64::MAX >> (64 - bits),
            pending: false,
        }))
    }
    pub fn begin(&mut self, command: vk::CommandBuffer) {
        unsafe {
            self.owner
                .device
                .cmd_reset_query_pool(command, self.pool, 0, 2);
            self.owner.device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.pool,
                0,
            );
        }
    }
    pub fn end(&mut self, command: vk::CommandBuffer) {
        unsafe {
            self.owner.device.cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                1,
            );
        }
        self.pending = true;
    }
    pub fn retire(&mut self) -> Result<(), VulkanError> {
        if !self.pending {
            return Ok(());
        }
        let mut ticks = [0u64; 2];
        // The slot timeline signal proves both writes completed. Never use WAIT.
        unsafe {
            self.owner.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(|e| VulkanError::vk("completed queue timestamps", e))?;
        let elapsed = ticks[1].wrapping_sub(ticks[0]) & self.mask;
        record(
            Counter::HostQueueGpuNanoseconds,
            (elapsed as f64 * self.period) as u64,
        );
        record(Counter::HostQueueTimedSubmissions, 1);
        self.pending = false;
        Ok(())
    }
}
impl Drop for Queries {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_query_pool(self.pool, None);
        }
    }
}
