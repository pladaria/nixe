//! Safe owner reads flush the actual buffer and texture caches before RAM access.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L350-L410
use super::*;
use nixe_memory::{AccessOperation, AccessResult, AllocationRange, DeviceAccessRequest};

impl Residency<'_> {
    pub(crate) fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        self.download_texture_memory(address, bytes)?;
        self.download_buffer_memory(address, bytes)?;
        self.caches
            .query_cache
            .lock()
            .unwrap()
            .flush_region(address, bytes)
    }

    pub(crate) fn progress_native_work(&mut self) -> Result<(), VulkanError> {
        self.runtime.device.poll_progress().map(|_| ())
    }

    pub(crate) fn queue_device_access(
        &mut self,
        mut request: DeviceAccessRequest,
    ) -> Result<(), VulkanError> {
        if request.is_cancelled() {
            request.finish(Err(Arc::new(nixe_memory::MemoryError::Execution(
                "GPU memory request was cancelled".into(),
            ))));
            return Ok(());
        }
        let result = (|| {
            self.runtime.device.realize_deferred_clear()?;
            let (ranges, operation) = request.parts_mut();
            self.access_memory(ranges, operation)
        })();
        match result {
            Ok(result) => {
                request.finish(Ok(result));
                Ok(())
            }
            Err(error) => {
                let cause = match &error {
                    VulkanError::Memory(cause) => Arc::clone(cause),
                    other => Arc::new(nixe_memory::MemoryError::Execution(
                        other.to_string().into(),
                    )),
                };
                request.finish(Err(cause));
                Err(error)
            }
        }
    }

    pub(crate) fn access_memory(
        &mut self,
        ranges: &[AllocationRange],
        operation: &mut AccessOperation,
    ) -> Result<AccessResult, VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        if let AccessOperation::Flush(area) = operation {
            if area.start >= area.end || area.end > 1 << self.runtime.device_memory.address_bits() {
                return Err(VulkanError::InvalidResource(
                    "invalid rasterizer flush area",
                ));
            }
            self.flush_region(area.start, area.end - area.start)?;
            return Ok(AccessResult::Flushed);
        }
        // CPU scalar reads use GetFlushArea above. Safe GPU-owner reads instead
        // flush both actual caches, as MemoryManager::ReadBlock does in Eden.
        if !matches!(operation, AccessOperation::Write(_)) {
            for extent in ranges.iter().flat_map(AllocationRange::physical_extents) {
                for (address, bytes) in self.runtime.device_memory.device_ranges(extent) {
                    self.download_buffer_memory(address, bytes)?;
                    self.download_texture_memory(address, bytes)?;
                    self.caches
                        .query_cache
                        .lock()
                        .unwrap()
                        .flush_region(address, bytes)?;
                }
            }
        }
        let writes = matches!(
            operation,
            AccessOperation::Write(_) | AccessOperation::Atomic(_)
        );
        let result = self
            .runtime
            .memory
            .apply_resolved_access(ranges, operation)?;
        if writes {
            for extent in ranges.iter().flat_map(AllocationRange::physical_extents) {
                for (address, bytes) in self.runtime.device_memory.device_ranges(extent) {
                    self.caches
                        .query_cache
                        .lock()
                        .unwrap()
                        .invalidate_region(address, bytes);
                    self.shader_cache.invalidate_region(address, bytes);
                    self.buffer_cache.write_memory(address, bytes);
                    self.texture_cache.write_memory(address, bytes);
                }
            }
        }
        Ok(result)
    }
}
