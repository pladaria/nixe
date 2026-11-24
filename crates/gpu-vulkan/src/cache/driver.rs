//! One driver pipeline cache shared by all construction workers for this title.
//! Vulkan internally synchronizes caches created without EXTERNALLY_SYNCHRONIZED.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp#L1047-L1138
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
const MAGIC: &[u8; 8] = b"NXVKDRV1";
pub(super) struct DriverCache {
    owner: Arc<DeviceOwner>,
    pub(super) handle: vk::PipelineCache,
    path: Option<PathBuf>,
}
impl DriverCache {
    pub fn new(
        device: &VulkanDevice,
        directory: Option<&std::path::Path>,
    ) -> Result<Self, VulkanError> {
        let started = Instant::now();
        let path = directory.map(|dir| dir.join("vulkan_pipelines.bin"));
        let mut initial = path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| {
                let profile = device.profile();
                let valid = bytes.len() >= 44
                    && &bytes[..8] == MAGIC
                    && bytes[8..12] == super::PERSISTENCE_VERSION.to_le_bytes()
                    && bytes[12..16] == 32u32.to_le_bytes()
                    && bytes[16..20] == 1u32.to_le_bytes()
                    && bytes[20..24] == profile.vendor_id.to_le_bytes()
                    && bytes[24..28] == profile.device_id.to_le_bytes()
                    && bytes[28..44] == profile.pipeline_cache_uuid;
                if valid {
                    Some(bytes)
                } else {
                    log::info!("ignored incompatible Vulkan driver pipeline cache");
                    None
                }
            });
        let create = |data: &[u8]| unsafe {
            device.owner.device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(data),
                None,
            )
        };
        let handle = match initial.as_deref() {
            Some(bytes) => create(&bytes[12..]).or_else(|error| {
                log::warn!(
                    "Vulkan driver cache rejected ({error:?}); starting with an empty cache"
                );
                initial = None;
                create(&[])
            }),
            None => create(&[]),
        }
        .map_err(|error| VulkanError::vk("vkCreatePipelineCache", error))?;
        if let Some(data) = &initial {
            log::info!(
                "loaded Vulkan driver pipeline cache: {} bytes in {:?} ({})",
                data.len() - 12,
                started.elapsed(),
                path.as_ref().unwrap().display()
            );
        } else if let Some(path) = &path {
            log::info!(
                "initialized empty Vulkan driver pipeline cache in {:?} ({})",
                started.elapsed(),
                path.display()
            );
        }
        Ok(Self {
            owner: Arc::clone(&device.owner),
            handle,
            path,
        })
    }
    pub fn checkpoint(&self) -> Result<usize, VulkanError> {
        let Some(path) = &self.path else {
            return Ok(0);
        };
        let bytes = unsafe { self.owner.device.get_pipeline_cache_data(self.handle) }
            .map_err(|error| VulkanError::vk("vkGetPipelineCacheData", error))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        }
        let write = || -> std::io::Result<()> {
            let mut tmp = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
            let mut file = BufWriter::new(tmp.as_file_mut());
            file.write_all(MAGIC)?;
            file.write_all(&super::PERSISTENCE_VERSION.to_le_bytes())?;
            file.write_all(&bytes)?;
            file.flush()?;
            drop(file);
            tmp.persist(path)?;
            Ok(())
        };
        if let Err(error) = write() {
            return Err(VulkanError::Compilation(error.to_string().into()));
        }
        Ok(bytes.len())
    }
}
impl Drop for DriverCache {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline_cache(self.handle, None);
        }
    }
}
