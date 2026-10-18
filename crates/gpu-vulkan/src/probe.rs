//! Optional native binding probe, excluded from the normal renderer build.
use crate::{BufferBinding, VulkanError, VulkanResidency, device::DeviceOwner};
use ash::vk;
use nixe_memory::{CanonicalAllocation, MemoryPermissions};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
pub(crate) struct ProbeQueries {
    owner: Arc<DeviceOwner>,
    handle: vk::QueryPool,
}
impl Drop for ProbeQueries {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_query_pool(self.handle, None);
        }
    }
}
#[derive(Debug)]
pub struct BindingMeasurement {
    pub binding: BufferBinding,
    pub cpu_ns_per_dispatch: f64,
    pub gpu_ns_per_dispatch: f64,
}
impl VulkanResidency {
    pub fn measure_bindings(
        &mut self,
        dispatches: u32,
        trials: u32,
    ) -> Result<Vec<BindingMeasurement>, VulkanError> {
        if dispatches == 0 || trials == 0 {
            return Err(VulkanError::InvalidResource(
                "binding probe counts must be nonzero",
            ));
        }
        let owner = Arc::clone(&self.device.owner);
        let family = self.device.queues().graphics_compute_family;
        let properties = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_queue_family_properties(owner.physical)
        };
        let bits = properties[family as usize].timestamp_valid_bits;
        if bits == 0 || self.device.profile().limits.timestamp_compute_and_graphics == 0 {
            return Err(VulkanError::InvalidResource(
                "binding probe needs graphics/compute queue timestamps",
            ));
        }
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mut measurements = Vec::new();
        for binding in [
            BufferBinding::DescriptorSet,
            BufferBinding::DeviceAddress,
            BufferBinding::DescriptorBuffer,
        ] {
            if binding == BufferBinding::DeviceAddress
                && (!self.device.profile().buffer_device_address
                    || self.device.profile().core.shader_int64 == 0)
                || binding == BufferBinding::DescriptorBuffer
                    && !self.device.profile().descriptor_buffer
            {
                continue;
            }
            let allocation = CanonicalAllocation::zeroed(4096, 4096)
                .map_err(|e| VulkanError::CanonicalMemory(e.to_string().into()))?;
            allocation
                .write(0, &0x44332211u32.to_ne_bytes().repeat(1024))
                .map_err(|e| VulkanError::CanonicalMemory(e.to_string().into()))?;
            let range = allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .map_err(|e| VulkanError::CanonicalMemory(e.to_string().into()))?;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | if binding == BufferBinding::DescriptorSet {
                    vk::BufferUsageFlags::empty()
                } else {
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                };
            let buffer = self.buffer_for(range, usage)?;
            // Compile and establish residency before either clock starts.
            self.swap_rgba8_words_binding(buffer, 0, 1024, binding)?;
            let work = self.flush()?.expect("warmup recording");
            self.device.wait_for_work(work, Duration::MAX)?;
            let mut cpu = Vec::new();
            let mut gpu = Vec::new();
            for _ in 0..trials {
                let handle = unsafe {
                    owner.device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                }
                .map_err(|e| VulkanError::vk("binding probe queries", e))?;
                let query = Arc::new(ProbeQueries {
                    owner: Arc::clone(&owner),
                    handle,
                });
                let index = self.device.begin_recording()?;
                self.device.submission.slots[index]
                    .probe_queries
                    .push(Arc::clone(&query));
                let command = self.device.submission.slots[index].command;
                unsafe {
                    self.device
                        .dispatch
                        .cmd_reset_query_pool(command, handle, 0, 2);
                    self.device.dispatch.timestamp(
                        command,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        handle,
                        0,
                    );
                }
                let start = Instant::now();
                for _ in 0..dispatches {
                    self.swap_rgba8_words_binding(buffer, 0, 1024, binding)?;
                }
                let elapsed = start.elapsed().as_nanos() as f64;
                unsafe {
                    self.device.dispatch.timestamp(
                        command,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        handle,
                        1,
                    );
                }
                let work = self.flush()?.expect("probe recording");
                self.device.wait_for_work(work, Duration::MAX)?;
                let mut ticks = [0u64; 2];
                unsafe {
                    owner.device.get_query_pool_results(
                        handle,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                }
                .map_err(|e| VulkanError::vk("binding probe results", e))?;
                cpu.push(elapsed / f64::from(dispatches));
                gpu.push(
                    ((ticks[1].wrapping_sub(ticks[0])) & mask) as f64
                        * f64::from(self.device.profile().limits.timestamp_period)
                        / f64::from(dispatches),
                );
            }
            cpu.sort_by(f64::total_cmp);
            gpu.sort_by(f64::total_cmp);
            let expected = if (1 + u64::from(dispatches) * u64::from(trials)).is_multiple_of(2) {
                0x44332211u32
            } else {
                0x44112233u32
            };
            if self.read_buffer(buffer, 0, 4096)? != expected.to_ne_bytes().repeat(1024) {
                return Err(VulkanError::InvalidResource(
                    "binding probe output mismatch",
                ));
            }
            measurements.push(BindingMeasurement {
                binding,
                cpu_ns_per_dispatch: cpu[cpu.len() / 2],
                gpu_ns_per_dispatch: gpu[gpu.len() / 2],
            });
        }
        Ok(measurements)
    }
}
