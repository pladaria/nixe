//! Bridge-owned Vulkan compilation data. Never shares wgpu's opaque cache.
use super::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

// Bump when the native translator/linkage/resource ABI changes. Vulkan also
// keys actual shader and pipeline state; this envelope is a compatibility gate,
// not a substitute pipeline key. Fixed little-endian encoding, no Rust layout.
const TRANSLATION_ABI: u32 = 1;
const IDENTITY_BYTES: usize = 44;
const HEADER_BYTES: usize = IDENTITY_BYTES + 8 + 32;

fn identity(properties: &vk::PhysicalDeviceProperties) -> [u8; IDENTITY_BYTES] {
    let mut bytes = Vec::with_capacity(IDENTITY_BYTES);
    bytes.extend_from_slice(b"NXVKPC01");
    for word in [
        TRANSLATION_ABI,
        usize::BITS
            | if cfg!(target_endian = "big") {
                0x100
            } else {
                0
            },
        properties.vendor_id,
        properties.device_id,
        properties.driver_version,
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&properties.pipeline_cache_uuid);
    bytes.try_into().unwrap()
}

fn valid_payload(data: &[u8], properties: &vk::PhysicalDeviceProperties) -> bool {
    // Cache headers are tightly packed little endian, not native C structs.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersionOne.html
    data.len() >= 32
        && data[0..4] == 32_u32.to_le_bytes()
        && data[4..8] == 1_u32.to_le_bytes()
        && data[8..12] == properties.vendor_id.to_le_bytes()
        && data[12..16] == properties.device_id.to_le_bytes()
        && data[16..32] == properties.pipeline_cache_uuid
}

fn load(
    path: &Path,
    properties: &vk::PhysicalDeviceProperties,
    bound: u64,
) -> std::io::Result<Vec<u8>> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let limit = bound.saturating_add(HEADER_BYTES as u64);
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "native pipeline cache is not a regular file",
        ));
    }
    if metadata.len() > limit {
        log::debug!(
            "ignoring oversized native pipeline cache: {}",
            path.display()
        );
        return Ok(Vec::new());
    }
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    let valid = bytes.len() >= HEADER_BYTES
        && bytes.len() as u64 <= limit
        && bytes[..IDENTITY_BYTES] == identity(properties)
        && bytes[IDENTITY_BYTES..IDENTITY_BYTES + 8]
            == ((bytes.len() - HEADER_BYTES) as u64).to_le_bytes()
        && bytes[IDENTITY_BYTES + 8..HEADER_BYTES] == Sha256::digest(&bytes[HEADER_BYTES..])[..]
        && valid_payload(&bytes[HEADER_BYTES..], properties);
    if !valid {
        // A discarded optimization is not an unsupported guest operation.
        log::debug!(
            "ignoring incompatible or damaged native pipeline cache: {}",
            path.display()
        );
        return Ok(Vec::new());
    }
    bytes.drain(..HEADER_BYTES);
    log::info!(
        "loaded native Vulkan pipeline cache: path={} bytes={}",
        path.display(),
        bytes.len()
    );
    Ok(bytes)
}

pub(super) struct NativePipelineCache {
    _device: Device,
    raw: ash::Device,
    handle: vk::PipelineCache,
    path: Option<PathBuf>,
    properties: vk::PhysicalDeviceProperties,
    bound: u64,
}

impl NativePipelineCache {
    pub(super) fn new(
        device: &Device,
        directory: Option<&Path>,
        bound: u64,
    ) -> Result<Self, BackendDriverError> {
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| unsupported("native pipeline cache needs Vulkan"))?;
        let raw = hal.raw_device().clone();
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_properties(hal.raw_physical_device())
        };
        let path = directory.map(|dir| {
            dir.join(format!(
                "native-vulkan-{:08x}-{:08x}.bin",
                properties.vendor_id, properties.device_id
            ))
        });
        let data = path
            .as_ref()
            .map(|path| {
                load(path, &properties, bound).map_err(|e| {
                    error(format!(
                        "cannot load native pipeline cache {}: {e}",
                        path.display()
                    ))
                })
            })
            .transpose()?
            .unwrap_or_default();
        // Private versioned/checksummed data produced by this driver's cache API.
        // No externally-synchronized flag: the driver supplies internal safety;
        // Nixe additionally confines mutation to its exclusive backend owner.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkCreatePipelineCache.html
        let handle = unsafe {
            raw.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(&data),
                None,
            )
        }
        .map_err(error)?;
        Ok(Self {
            _device: device.clone(),
            raw,
            handle,
            path,
            properties,
            bound,
        })
    }

    pub(super) fn handle(&mut self) -> vk::PipelineCache {
        self.handle
    }

    fn data(&self) -> Result<Vec<u8>, BackendDriverError> {
        // ash's convenience helper allocates the entire driver-reported size.
        // Query directly to enforce the configured bound *before* allocation.
        // VK_INCOMPLETE returns a valid cache subset, not arbitrarily cut bytes.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineCacheData.html
        let query = self.raw.fp_v1_0().get_pipeline_cache_data;
        let mut size = 0;
        unsafe {
            query(
                self.raw.handle(),
                self.handle,
                &mut size,
                std::ptr::null_mut(),
            )
        }
        .result()
        .map_err(error)?;
        size = size.min(usize::try_from(self.bound).unwrap_or(usize::MAX));
        if size < 32 {
            return Ok(Vec::new());
        }
        let mut bytes = vec![0_u8; size];
        let status = unsafe {
            query(
                self.raw.handle(),
                self.handle,
                &mut size,
                bytes.as_mut_ptr().cast(),
            )
        };
        if status != vk::Result::SUCCESS && status != vk::Result::INCOMPLETE {
            return Err(error(status));
        }
        bytes.truncate(size);
        Ok(bytes)
    }

    pub(super) fn persist(&self) -> Result<(), BackendDriverError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let data = self.data()?;
        if data.is_empty() {
            return Ok(());
        }
        if !valid_payload(&data, &self.properties) {
            return Err(error(
                "driver returned an invalid native pipeline cache header",
            ));
        }
        let directory = path.parent().unwrap();
        std::fs::create_dir_all(directory).map_err(|e| {
            error(format!(
                "cannot create native cache directory {}: {e}",
                directory.display()
            ))
        })?;
        // Unique sibling + atomic replacement also supports concurrent instances.
        // Never remove the old cache before publishing the new complete file.
        let mut file = tempfile::NamedTempFile::new_in(directory).map_err(|e| {
            error(format!(
                "cannot stage native pipeline cache {}: {e}",
                path.display()
            ))
        })?;
        file.write_all(&identity(&self.properties))
            .and_then(|()| file.write_all(&(data.len() as u64).to_le_bytes()))
            .and_then(|()| file.write_all(&Sha256::digest(&data)))
            .and_then(|()| file.write_all(&data))
            .and_then(|()| file.as_file().sync_all())
            .map_err(|e| {
                error(format!(
                    "cannot write native pipeline cache {}: {e}",
                    path.display()
                ))
            })?;
        file.persist(path).map_err(|e| {
            error(format!(
                "cannot publish native pipeline cache {}: {e}",
                path.display()
            ))
        })?;
        log::info!(
            "saved native Vulkan pipeline cache: path={} bytes={}",
            path.display(),
            data.len()
        );
        Ok(())
    }
}

impl Drop for NativePipelineCache {
    fn drop(&mut self) {
        // Pipeline-cache lifetime is independent of compiled pipelines/GPU work.
        unsafe {
            self.raw.destroy_pipeline_cache(self.handle, None);
        }
    }
}

#[cfg(test)]
#[path = "persistent_cache_tests.rs"]
mod tests;
