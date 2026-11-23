//! Native address tables retain the physical targets of indirect shader accesses.
use super::*;
use nixe_gpu::native::GuestBufferRange;

pub struct PointerTableBinding {
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

impl Residency<'_> {
    pub(crate) fn nvn_pointer_table(
        &mut self,
        range: nixe_nvn::Range,
        start: i32,
        end: i32,
        written: bool,
    ) -> Result<super::execution::HostBinding, VulkanError> {
        use super::execution::HostBinding;
        if range.address == 0 || range.size == 0 {
            return Ok(HostBinding::NullBuffer);
        }
        let limits = &self.device().profile().limits;
        let alignment = limits.min_storage_buffer_offset_alignment.max(8);
        let padding = range.address % alignment;
        if !range.address.is_multiple_of(8)
            || !range.size.is_multiple_of(8)
            || range.size + padding > u64::from(limits.max_storage_buffer_range)
        {
            return Err(VulkanError::InvalidResource(
                "invalid NVN pointer table range",
            ));
        }
        self.flush_read_span(range.address, range.size)?;
        let mut words = smallvec::SmallVec::<[u64; 32]>::from_elem(0, range.size as usize / 8);
        self.runtime
            .device_reader
            .read_block_unsafe(range.address, bytemuck::cast_slice_mut(&mut words))?;

        let mut aliases = smallvec::SmallVec::<[GuestBufferRange; 4]>::new();
        if written {
            self.device_memory().apply_device_aliases(
                range.address,
                range.size,
                |address, size| {
                    aliases.push(GuestBufferRange { address, size });
                },
            );
            aliases.push(GuestBufferRange {
                address: range.address,
                size: range.size,
            });
        }
        let size = (i64::from(end) - i64::from(start)) as u64;
        let mut targets =
            smallvec::SmallVec::<[Option<GuestBufferRange>; 32]>::with_capacity(words.len());
        for word in words {
            let pointer = u64::from_le(word);
            if pointer == 0 {
                targets.push(None);
                continue;
            }
            let address = pointer
                .checked_add_signed(i64::from(start))
                .filter(|address| address.is_multiple_of(4))
                .ok_or(VulkanError::InvalidResource(
                    "invalid NVN pointer table target",
                ))?;
            if aliases
                .iter()
                .any(|alias| address < alias.address + alias.size && alias.address < address + size)
            {
                return Err(VulkanError::InvalidResource(
                    "NVN shader writes alias their pointer table",
                ));
            }
            targets.push(Some(GuestBufferRange { address, size }));
        }
        self.pointer_table(&targets, padding, start)
            .map(|table| HostBinding::PointerTable(Arc::new(table)))
    }
}
