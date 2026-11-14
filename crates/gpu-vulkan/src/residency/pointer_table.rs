//! Native address tables retain the physical targets of indirect shader accesses.
use super::*;
use nixe_gpu::native::GuestBufferRange;

pub(crate) struct PointerTableBinding {
    pub(super) table: Arc<AllocatedBuffer>,
    pub(super) offset: u64,
    pub(super) size: u64,
    pub(super) targets: Box<[BufferRange]>,
}

impl Residency<'_> {
    pub(crate) fn pointer_table(
        &mut self,
        targets: &[Option<GuestBufferRange>],
        padding: u64,
        start: i32,
    ) -> Result<PointerTableBinding, VulkanError> {
        if !self.runtime.device.profile().buffer_device_address {
            return Err(VulkanError::InvalidResource(
                "indirect pointer tables require buffer device addresses",
            ));
        }
        let pinned = self.runtime.constructing.len();
        let result = (|| {
            // Resolve all allocations before deriving addresses: later overlapping
            // targets can merge and retire a buffer selected earlier in the list.
            let mut spans: Vec<_> = targets.iter().copied().flatten().collect();
            spans.sort_unstable_by_key(|r| r.address);
            let mut merged: Vec<GuestBufferRange> = Vec::new();
            for span in spans {
                if let Some(last) = merged.last_mut()
                    && span.address <= last.address + last.size
                {
                    last.size = last.size.max(span.address + span.size - last.address);
                } else {
                    merged.push(span);
                }
            }
            for span in &merged {
                let range = self.device_buffer(
                    span.address,
                    span.size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    false,
                    false,
                )?;
                self.runtime
                    .constructing
                    .push(Resource::Buffer(range.buffer));
            }
            let mut ranges = Vec::with_capacity(merged.len());
            for span in &merged {
                let range = self.device_buffer(
                    span.address,
                    span.size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    true,
                    false,
                )?;
                self.runtime
                    .constructing
                    .push(Resource::Buffer(range.buffer));
                self.use_buffer(range.buffer, range.offset, range.size)?;
                ranges.push(range);
            }
            let mut addresses = HashMap::default();
            for range in &ranges {
                addresses.entry(range.buffer).or_insert_with(|| unsafe {
                    self.runtime.device.owner.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::default()
                            .buffer(self.buffer_cache.get(range.buffer).native.handle),
                    )
                });
            }
            let mut bytes = vec![0u8; padding as usize + targets.len() * 8];
            for (target, bytes) in targets
                .iter()
                .zip(bytes[padding as usize..].chunks_exact_mut(8))
            {
                let Some(target) = target else { continue };
                let index = merged.partition_point(|range| range.address <= target.address) - 1;
                let range = ranges[index];
                let base = addresses[&range.buffer];
                let native = base + range.offset + (target.address - merged[index].address);
                let native = native.checked_add_signed(-i64::from(start)).ok_or(
                    VulkanError::InvalidResource("native indirect pointer overflow"),
                )?;
                bytes.copy_from_slice(&native.to_le_bytes());
            }
            let alignment = self
                .runtime
                .device
                .profile()
                .limits
                .min_storage_buffer_offset_alignment
                .max(8);
            let slice = self
                .runtime
                .device
                .submission
                .reserve_upload_aligned(&bytes, alignment)?;
            self.runtime.statistics.cpu_streamed_bytes += bytes.len() as u64;
            Ok(PointerTableBinding {
                table: Arc::clone(self.runtime.device.submission.staging_buffer(&slice)),
                offset: slice.offset,
                size: bytes.len() as u64,
                targets: ranges.into_boxed_slice(),
            })
        })();
        self.runtime.constructing.truncate(pinned);
        result
    }
}
