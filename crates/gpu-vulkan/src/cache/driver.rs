//! Per-worker VkPipelineCache objects: compilation never shares an unsynchronized
//! cache or serializes all workers on one driver lock. Checkpoint is explicit.
use super::{CacheConfiguration, CompileError, persistent};
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use nixe_gpu::shader_identity::ContentId;
use std::sync::{Arc, Mutex};
struct Cache {
    owner: Arc<DeviceOwner>,
    handle: vk::PipelineCache,
}
impl Drop for Cache {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline_cache(self.handle, None);
        }
    }
}
pub(super) struct DriverCaches {
    caches: Vec<Mutex<Cache>>,
    store: Option<persistent::Store>,
    key: ContentId,
    header: [u8; 32],
    maximum: u64,
}
impl DriverCaches {
    pub fn new(device: &VulkanDevice, config: &CacheConfiguration) -> Result<Self, VulkanError> {
        let profile = device.profile();
        let mut header = [0; 32];
        header[..4].copy_from_slice(&32u32.to_le_bytes());
        header[4..8].copy_from_slice(&1u32.to_le_bytes());
        header[8..12].copy_from_slice(&profile.vendor_id.to_le_bytes());
        header[12..16].copy_from_slice(&profile.device_id.to_le_bytes());
        header[16..].copy_from_slice(&profile.pipeline_cache_uuid);
        let key = ContentId::of(
            b"nixe-vulkan-driver-cache-v1",
            &[&header, &profile.driver_version.to_le_bytes()],
        );
        let store = persistent::store(
            config.directory.as_deref(),
            "driver-v1",
            config.persistent_bytes,
        );
        let initial = store
            .as_ref()
            .and_then(|s| s.read(key))
            .filter(|bytes| valid_header(bytes, &header));
        let mut result = Self {
            caches: Vec::new(),
            store,
            key,
            header,
            maximum: config.persistent_bytes,
        };
        for _ in 0..config.workers {
            let handle = unsafe {
                device.owner.device.create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo::default()
                        .initial_data(initial.as_deref().unwrap_or(&[])),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("worker pipeline cache", e))?;
            result.caches.push(Mutex::new(Cache {
                owner: Arc::clone(&device.owner),
                handle,
            }));
        }
        Ok(result)
    }
    pub fn with<T>(
        &self,
        worker: usize,
        create: impl FnOnce(vk::PipelineCache) -> Result<T, CompileError>,
    ) -> Result<T, CompileError> {
        let cache = self.caches[worker].lock().unwrap();
        create(cache.handle)
    }
    /// Called outside recording/control servicing. Locks also exclude worker
    /// creation while Vulkan merges/serializes its externally synchronized cache.
    pub fn checkpoint(&self) -> Result<(), CompileError> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        let caches: Vec<_> = self
            .caches
            .iter()
            .map(|cache| cache.lock().unwrap())
            .collect();
        let first = &caches[0];
        let sources: Vec<_> = caches.iter().skip(1).map(|cache| cache.handle).collect();
        if !sources.is_empty() {
            unsafe {
                first
                    .owner
                    .device
                    .merge_pipeline_caches(first.handle, &sources)
            }
            .map_err(|e| VulkanError::vk("merge worker pipeline caches", e))?;
        }
        let bytes = unsafe { first.owner.device.get_pipeline_cache_data(first.handle) }
            .map_err(|e| VulkanError::vk("serialize driver cache", e))?;
        if bytes.len() as u64 <= self.maximum && valid_header(&bytes, &self.header) {
            store
                .write(self.key, &bytes)
                .map_err(|e| CompileError::Failed(e.to_string().into()))?;
        }
        Ok(())
    }
}
fn valid_header(bytes: &[u8], expected: &[u8; 32]) -> bool {
    // Vulkan specifies little-endian version-one fields independent of the host.
    // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-cache-header
    bytes.len() >= 32 && bytes[..32] == *expected
}
