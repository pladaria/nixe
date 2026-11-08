//! The shared GPU page table used directly by the command processor.
use super::*;
use std::sync::{OnceLock, RwLock, Weak, atomic::AtomicU64};

/// MemoryManager::BindRasterizer connects GPU mapping changes to the actual
/// buffer and texture cache owners. It is independent of device RAM coherence.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/rasterizer_interface.h
pub trait MaxwellGpuMemoryInterface: Send + Sync + std::fmt::Debug {
    fn modify_gpu_memory(&self, address_space: MaxwellAddressSpaceId, address: u64, bytes: u64);
    fn unmap_memory(&self, address: u64, bytes: u64);
}

#[derive(Debug)]
pub(super) struct TranslationState {
    id: MaxwellAddressSpaceId,
    machine: Machine,
    device: crate::MaxwellDeviceMemory,
    profile: MaxwellGpuProfile,
    pub(super) table: PageTable,
    pub(super) mappings: RwLock<Mappings>,
    pub(super) generation: AtomicU64,
    pub(super) initialized: AtomicBool,
    rasterizer: OnceLock<Weak<dyn MaxwellGpuMemoryInterface>>,
}
#[derive(Debug)]
pub(super) struct Mappings {
    pub(super) kinds: super::kinds::KindRanges,
    pub(super) live: BTreeMap<GpuVirtualAddress, LiveMapping>,
}
/// Kernel and ingress handles share the live GPU page table, never GPU-owned state.
#[derive(Clone, Debug)]
pub struct MaxwellGpuMemoryHandle {
    pub(super) inner: Arc<TranslationState>,
}
/// The graphics owner has one memory manager per address space. Its invalidation
/// accumulator and scratch ranges persist across channels and command lists.
#[derive(Debug)]
pub struct MaxwellGpuMemory {
    pub(super) inner: Arc<TranslationState>,
    device_reader: nixe_memory::device_memory::DeviceMemoryReader,
    invalidations: super::invalidation::InvalidationAccumulator,
    invalidation_regions: Vec<nixe_gpu::native::GuestBufferRange>,
}

impl MaxwellGpuMemoryHandle {
    pub(super) fn new(
        device: crate::MaxwellDeviceMemory,
        id: MaxwellAddressSpaceId,
        profile: MaxwellGpuProfile,
    ) -> Self {
        Self {
            inner: Arc::new(TranslationState {
                id,
                machine: device.machine().clone(),
                table: PageTable::new(
                    device.machine(),
                    profile.virtual_address().address_bits().bits(),
                ),
                device,
                profile,
                mappings: RwLock::new(Mappings {
                    kinds: super::kinds::KindRanges::new(
                        profile.virtual_address().address_bits().bits(),
                    ),
                    live: BTreeMap::new(),
                }),
                generation: AtomicU64::new(MappingGeneration::INITIAL.get()),
                initialized: AtomicBool::new(false),
                rasterizer: OnceLock::new(),
            }),
        }
    }
    pub fn id(&self) -> MaxwellAddressSpaceId {
        self.inner.id
    }
    pub fn initialized(&self) -> bool {
        self.inner.initialized.load(Ordering::Acquire)
    }
    pub fn machine(&self) -> &Machine {
        &self.inner.machine
    }
    pub fn device_memory(&self) -> &crate::MaxwellDeviceMemory {
        &self.inner.device
    }
    pub fn bind_rasterizer(&self, rasterizer: Weak<dyn MaxwellGpuMemoryInterface>) {
        self.inner
            .rasterizer
            .set(rasterizer)
            .expect("GPU memory rasterizer already bound");
    }
    /// Notify after releasing the live mapping registry. Cache consumers may
    /// resolve cold physical inputs while holding their own content locks.
    pub(super) fn modify_gpu_memory(&self, address: u64, bytes: u64) {
        if bytes != 0
            && let Some(rasterizer) = self.inner.rasterizer.get().and_then(Weak::upgrade)
        {
            rasterizer.modify_gpu_memory(self.inner.id, address, bytes);
        }
    }
    /// The old live mapping still owns the device pin during this callback.
    /// Release the GPU registry writer first: cache preparation can resolve
    /// cold physical inputs while holding the actual cache locks.
    pub(super) fn unmap_memory(&self, live: &LiveMapping, start: u64, end: u64) {
        let mapping = &live.descriptor;
        let begin = start.max(mapping.offset.get());
        let end = end.min(mapping.offset.get() + mapping.size);
        if begin < end
            && let Some(rasterizer) = self.inner.rasterizer.get().and_then(Weak::upgrade)
        {
            let address =
                live._device.address() + mapping.backing_offset + begin - mapping.offset.get();
            rasterizer.unmap_memory(address, end - begin);
        }
    }
}
impl MaxwellGpuMemory {
    pub fn new(handle: MaxwellGpuMemoryHandle) -> Self {
        let device_reader = handle.device_memory().device().reader();
        Self {
            inner: handle.inner,
            device_reader,
            invalidations: Default::default(),
            invalidation_regions: Vec::new(),
        }
    }
    fn live_mapping(mappings: &Mappings, address: GpuVirtualAddress) -> Option<&LiveMapping> {
        mappings
            .live
            .range(..=address)
            .next_back()
            .map(|(_, live)| live)
            .filter(|live| address.get() < live.descriptor.offset.get() + live.descriptor.size)
    }
    pub fn machine(&self) -> &Machine {
        &self.inner.machine
    }
    pub fn device_memory(&self) -> &crate::MaxwellDeviceMemory {
        &self.inner.device
    }
    pub fn mapping_generation(&self) -> MappingGeneration {
        MappingGeneration::new(self.inner.generation.load(Ordering::Acquire))
    }
    #[must_use]
    pub fn id(&self) -> MaxwellAddressSpaceId {
        self.inner.id
    }

    #[must_use]
    pub fn profile(&self) -> MaxwellGpuProfile {
        self.inner.profile
    }

    #[must_use]
    pub fn profile_id(&self) -> GpuProfileId {
        self.inner.profile.id()
    }

    #[must_use]
    pub fn initialized(&self) -> bool {
        self.inner.initialized.load(Ordering::Acquire)
    }

    /// Constructs an address using this address space's immutable profile
    /// width.
    pub fn address(&self, value: u64) -> Result<GpuVirtualAddress, GpuVirtualAddressError> {
        GpuVirtualAddress::try_new(
            value,
            self.inner.profile.virtual_address().address_bits().bits(),
        )
    }

    /// Adds a byte offset using this address space's immutable profile width.
    pub fn checked_add(
        &self,
        address: GpuVirtualAddress,
        byte_offset: u64,
    ) -> Result<GpuVirtualAddress, GpuVirtualAddressError> {
        address.checked_add(
            byte_offset,
            self.inner.profile.virtual_address().address_bits().bits(),
        )
    }

    /// The current hardware binding translates its GPU base once, as in
    /// Eden MemoryManager::GpuToCpuAddress. The live map owns its SMMU pin.
    #[inline]
    pub fn device_address(&self, offset: GpuVirtualAddress) -> Result<u64, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset)?;
        self.inner
            .table
            .translate(offset.get())
            .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: offset })
    }
    /// MemoryManager::GetPointer resolves the GPU and device PTEs directly.
    /// The returned numeric pointer is neither a physical lease nor a cache flush.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L266-L281
    #[inline]
    pub fn pointer(
        &self,
        address: GpuVirtualAddress,
        bytes: u64,
    ) -> Option<nixe_memory::HostAddress> {
        if !self.initialized() {
            return None;
        }
        self.inner
            .device
            .device()
            .pointer(self.inner.table.translate(address.get())?, bytes)
    }
    /// MemoryManager::GetSpan checks GPU-to-device continuity, then the device
    /// PTE's physical continuity. The caller owns the machine while consuming
    /// the numeric span, and resolves coherence separately before reading.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L721-L742
    pub(crate) fn get_span(
        &self,
        address: GpuVirtualAddress,
        bytes: u64,
    ) -> Result<Option<nixe_memory::HostAddress>, MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        if bytes == 0 {
            return Ok(None);
        }
        self.checked_add(address, bytes - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        if self.max_continuous_range(address, bytes)? != bytes {
            return Ok(None);
        }
        let Some(device) = self.inner.table.translate(address.get()) else {
            return Ok(None);
        };
        Ok(self.inner.device.device().span(device, bytes))
    }
    /// Safe GuestMemory reads flush actual device cache producers after GetSpan.
    /// No mapping snapshot, physical pin or command-source cache is involved.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L442-L462
    pub(crate) fn flush_region(
        &self,
        address: GpuVirtualAddress,
        bytes: u64,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        if bytes == 0 {
            return Ok(());
        }
        self.checked_add(address, bytes - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mut offset = 0;
        while offset < bytes {
            let span = self.inner.table.span(address.get() + offset);
            let count = span.bytes.min(bytes - offset);
            if let Some(device) = span.address {
                access
                    .flush_region(device, count)
                    .map_err(MaxwellGpuAccessError::Backing)?;
            }
            offset += count;
        }
        Ok(())
    }
    #[cold]
    fn scalar_read_error(&self, address: GpuVirtualAddress) -> MaxwellGpuAccessError {
        match self.device_address(address) {
            Err(error) => error,
            Ok(_) => MaxwellGpuAccessError::Backing(Arc::new(MemoryError::Invalid(
                "GPU scalar read has no device RAM backing",
            ))),
        }
    }
    /// MemoryManager::Read<T> translates the scalar's base and loads through
    /// GetPointer. Handles and NVN pointer descriptors do not use ReadBlock's
    /// page walker, temporary bytes or translation-slot lookup.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L232-L242
    #[inline]
    pub fn read_u32(&self, address: GpuVirtualAddress) -> Result<u32, MaxwellGpuAccessError> {
        let pointer = self
            .pointer(address, 4)
            .ok_or_else(|| self.scalar_read_error(address))?;
        // SAFETY: GetPointer bounds the complete scalar in the machine's
        // retained arena; raw instructions permit guest races without references.
        Ok(unsafe { pointer.read_u32(0) })
    }
    #[inline]
    pub fn read_u64(&self, address: GpuVirtualAddress) -> Result<u64, MaxwellGpuAccessError> {
        let pointer = self
            .pointer(address, 8)
            .ok_or_else(|| self.scalar_read_error(address))?;
        // SAFETY: as Read<u32>, with the eight-byte range checked by GetPointer.
        Ok(unsafe { pointer.read_u64(0) })
    }
    /// Consecutive GPU mappings are one span only when their DAddrs continue.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp (MaxContinuousRange)
    pub fn max_continuous_range(
        &self,
        address: GpuVirtualAddress,
        size: u64,
    ) -> Result<u64, MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        let mut current = address.get();
        let end = current
            .saturating_add(size)
            .min(1u64 << self.profile().virtual_address().address_bits().bits());
        let mut expected = None;
        while current < end {
            let span = self.inner.table.span(current);
            let Some(device) = span.address else {
                break;
            };
            if expected.is_some_and(|next| next != device) {
                break;
            }
            let count = span.bytes.min(end - current);
            current += count;
            expected = Some(device + count);
        }
        Ok(current - address.get())
    }
    /// GetPageKind has its own cold range-map exclusion in the reference.
    pub fn page_kind(&self, address: GpuVirtualAddress) -> Result<u8, MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        Ok(self.inner.mappings.read().unwrap().kinds.get(address.get()))
    }
    /// Same-kind layout continuity, independent of SMMU or current GPU mapping
    /// coverage, as Eden MemoryManager::GetMemoryLayoutSize.
    pub fn memory_layout_size(
        &self,
        address: GpuVirtualAddress,
    ) -> Result<u64, MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(mappings.kinds.continuous_size(address.get()))
    }
    /// Resolves the maximal active mapping suffix beginning at `offset`.
    pub fn resolve_maximal(
        &self,
        offset: GpuVirtualAddress,
    ) -> Result<MaxwellResolvedMapping, MaxwellGpuAccessError> {
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.resolve_maximal_locked(offset, &mappings)
    }

    fn resolve_maximal_locked(
        &self,
        offset: GpuVirtualAddress,
        mappings: &Mappings,
    ) -> Result<MaxwellResolvedMapping, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset)?;
        let mapping = &Self::live_mapping(mappings, offset)
            .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: offset })?
            .descriptor;
        let within_mapping = offset
            .get()
            .checked_sub(mapping.offset.get())
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        let size = mapping
            .size
            .checked_sub(within_mapping)
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        let backing_offset = mapping
            .backing_offset
            .checked_add(within_mapping)
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        Ok(MaxwellResolvedMapping {
            gpu_offset: offset,
            size,
            backing_offset,
            mapping: mapping.clone(),
        })
    }
    /// Resolves a complete GPU range into ordered retained mapping segments.
    ///
    /// Resolution is atomic: a hole, permission failure, or arithmetic error
    /// returns no partially usable range.
    pub fn resolve_range(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
    ) -> Result<MaxwellResolvedRange, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset)?;
        if size == 0 {
            return Err(MaxwellGpuAccessError::EmptyRange);
        }
        self.checked_add(offset, size - 1)
            .map_err(MaxwellGpuAccessError::Address)?;

        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.resolve_range_locked(offset, size, &mappings)
    }
    fn resolve_range_locked(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
        mappings: &Mappings,
    ) -> Result<MaxwellResolvedRange, MaxwellGpuAccessError> {
        let mut segments = Vec::new();
        self.resolve_segments_locked(offset, size, mappings, &mut segments)?;
        Ok(MaxwellResolvedRange {
            address_space: self.inner.id,
            offset,
            size,
            segments: segments.into(),
        })
    }
    /// GetSubmappedRange groups consecutive GPU pages by consecutive DAddr,
    /// without constructing physical mappings or retained range snapshots.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L647-L704
    pub fn get_submapped_range(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
    ) -> Result<smallvec::SmallVec<[(GpuVirtualAddress, u64); 8]>, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset)?;
        if size == 0 {
            return Err(MaxwellGpuAccessError::EmptyRange);
        }
        self.checked_add(offset, size - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mut ranges = smallvec::SmallVec::<[(GpuVirtualAddress, u64); 8]>::new();
        let mut cursor = offset.get();
        let end = cursor + size;
        let mut expected_device = None;
        while cursor < end {
            let span = self.inner.table.span(cursor);
            let bytes = span.bytes.min(end - cursor);
            if let Some(device) = span.address {
                if let Some((_, size)) = ranges.last_mut().filter(|(base, size)| {
                    base.get() + *size == cursor && expected_device == Some(device)
                }) {
                    *size += bytes;
                } else {
                    ranges.push((self.address(cursor).expect("validated GPU range"), bytes));
                }
                expected_device = Some(device + bytes);
            } else {
                expected_device = None;
            }
            cursor += bytes;
        }
        Ok(ranges)
    }

    fn resolve_segments_locked(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
        mappings: &Mappings,
        segments: &mut impl Extend<MaxwellResolvedMapping>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let mut remaining = size;
        let mut cursor = offset;
        while remaining != 0 {
            let mut segment = self.resolve_maximal_locked(cursor, mappings)?;
            segment.size = segment.size.min(remaining);
            remaining -= segment.size;
            if remaining != 0 {
                cursor = self
                    .checked_add(cursor, segment.size)
                    .map_err(MaxwellGpuAccessError::Address)?;
            }
            segments.extend(Some(segment));
        }
        Ok(())
    }

    /// WriteBlockCached changes RAM now and defers GPU cache invalidation to
    /// FlushCaching. No backend callback or native upload runs on a CB write.
    pub fn write_block_cached(
        &mut self,
        address: GpuVirtualAddress,
        bytes: &[u8],
    ) -> Result<(), MaxwellGpuAccessError> {
        self.write_cached(address, nixe_memory::WriteSource::Bytes(bytes))
    }
    pub(crate) fn write_cached(
        &mut self,
        address: GpuVirtualAddress,
        source: nixe_memory::WriteSource<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let bytes = source.len() as u64;
        self.write_upload(address, source)?;
        if bytes != 0 {
            self.invalidations.add(address.get(), bytes);
        }
        Ok(())
    }
    pub(crate) fn write_inline_words(
        &self,
        address: GpuVirtualAddress,
        source: crate::MaxwellDmaArguments<'_>,
        bytes: usize,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.write_inline_source(address, source.write_source().slice(0..bytes), access)
    }
    pub fn write_inline(
        &self,
        address: GpuVirtualAddress,
        bytes: &[u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.write_inline_source(address, nixe_memory::WriteSource::Bytes(bytes), access)
    }
    fn write_inline_source(
        &self,
        address: GpuVirtualAddress,
        source: nixe_memory::WriteSource<'_>,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let bytes = source.len() as u64;
        self.write_upload(address, source)?;
        let mut regions = Vec::new();
        Self::submapped_ranges(&self.inner.table, address.get(), bytes, &mut regions);
        access
            .inline_memory(&regions)
            .map_err(MaxwellGpuAccessError::Backing)
    }
    fn write_upload(
        &self,
        address: GpuVirtualAddress,
        source: nixe_memory::WriteSource<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        if source.is_empty() {
            return Ok(());
        }
        self.checked_add(address, source.len() as u64 - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mut at = address.get();
        let end = at + source.len() as u64;
        while at < end {
            let span = self.inner.table.span(at);
            let count = span.bytes.min(end - at);
            // WriteBlockImpl advances over reserved/unmapped GPU pages. Already
            // written prefixes are not preflighted, captured or rolled back.
            if let Some(device) = span.address {
                let begin = (at - address.get()) as usize;
                let source = source.slice(begin..begin + count as usize);
                let device_memory = self.device_memory().device();
                if span.continuous {
                    device_memory.write_contiguous(device, source)
                } else {
                    device_memory.write_block_unsafe(device, source)
                }
                .map_err(|e| MaxwellGpuAccessError::Backing(Arc::new(e)))?;
            }
            at += count;
        }
        Ok(())
    }

    fn submapped_ranges(
        table: &PageTable,
        address: u64,
        bytes: u64,
        ranges: &mut Vec<nixe_gpu::native::GuestBufferRange>,
    ) {
        let mut cursor = address;
        let end = address + bytes;
        let mut segment: Option<nixe_gpu::native::GuestBufferRange> = None;
        while cursor < end {
            let span = table.span(cursor);
            let size = span.bytes.min(end - cursor);
            if let Some(device) = span.address {
                if let Some(last) = segment
                    .as_mut()
                    .filter(|last| last.address + last.size == device)
                {
                    last.size += size;
                } else {
                    if let Some(last) = segment.take() {
                        ranges.push(last);
                    }
                    segment = Some(nixe_gpu::native::GuestBufferRange {
                        address: device,
                        size,
                    });
                }
            } else if let Some(last) = segment.take() {
                ranges.push(last);
            }
            cursor += size;
        }
        if let Some(last) = segment {
            ranges.push(last);
        }
    }
    /// MemoryManager::FlushCaching resolves the accumulated GPUVA against the
    /// current page table, then invalidates actual caches before taking draw locks.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L711-L719
    pub fn flush_caching(
        &mut self,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        if self.invalidations.is_empty() {
            return Ok(());
        }
        self.invalidation_regions.clear();
        let changed = self.invalidations.invalidate_all(|address, bytes| {
            Self::submapped_ranges(
                &self.inner.table,
                address,
                bytes,
                &mut self.invalidation_regions,
            );
        });
        if changed {
            access
                .invalidate_regions(&self.invalidation_regions)
                .map_err(MaxwellGpuAccessError::Backing)?;
            self.invalidation_regions.clear();
        }
        Ok(())
    }
    /// ReadBlock flushes the actual DAddr caches, then copies RAM directly.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
    pub fn read_into(
        &self,
        address: GpuVirtualAddress,
        output: &mut [u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.read_block(address, output, Some(access))
    }
    /// ReadBlockUnsafe consumes QMD, descriptors and CBuf RAM without cache flushes.
    pub fn read_into_unsafe(
        &self,
        address: GpuVirtualAddress,
        output: &mut [u8],
    ) -> Result<(), MaxwellGpuAccessError> {
        self.read_block(address, output, None)
    }
    fn read_block(
        &self,
        address: GpuVirtualAddress,
        output: &mut [u8],
        mut access: Option<&mut nixe_gpu::MemoryAccess<'_>>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.validate_resolution_request(address)?;
        if output.is_empty() {
            return Ok(());
        }
        self.checked_add(address, output.len() as u64 - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mut offset = 0;
        while offset < output.len() {
            let span = self.inner.table.span(address.get() + offset as u64);
            let count = (span.bytes as usize).min(output.len() - offset);
            let target = &mut output[offset..offset + count];
            if let Some(device) = span.address {
                if let Some(access) = access.as_deref_mut() {
                    access
                        .flush_region(device, count as u64)
                        .map_err(MaxwellGpuAccessError::Backing)?;
                }
                if span.continuous {
                    self.device_reader.read_contiguous(device, target)
                } else {
                    self.device_reader.read_block_unsafe(device, target)
                }
                .map_err(|error| MaxwellGpuAccessError::Backing(Arc::new(error)))?;
            } else {
                target.fill(0);
            }
            offset += count;
        }
        Ok(())
    }
    /// Pin exactly the accepted physical spans, without retaining adjacent
    /// nvmap bytes. Later operations use these pins and never retranslate a VA.
    pub fn capture(
        &self,
        resolved: &MaxwellResolvedRange,
    ) -> Result<Vec<AllocationRange>, MaxwellGpuAccessError> {
        let _change = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.validate_resolved(resolved)?;
        self.capture_validated(resolved)
    }
    // The caller holds the mapping read lock and validates the complete resolution.
    // Share exact slow-operation capture without recursively acquiring RwLock.
    fn capture_validated(
        &self,
        resolved: &MaxwellResolvedRange,
    ) -> Result<Vec<AllocationRange>, MaxwellGpuAccessError> {
        let mut ranges = Vec::new();
        for segment in resolved.segments.iter() {
            ranges.extend(
                segment
                    .mapping
                    .capture(segment.backing_offset, segment.size)
                    .map_err(|error| MaxwellGpuAccessError::Backing(Arc::new(error)))?,
            );
        }
        Ok(ranges)
    }
    fn validate_resolved(
        &self,
        resolved: &MaxwellResolvedRange,
    ) -> Result<(), MaxwellGpuAccessError> {
        if resolved.address_space != self.inner.id {
            return Err(MaxwellGpuAccessError::WrongAddressSpace {
                expected: resolved.address_space,
                actual: self.inner.id,
            });
        }
        for segment in resolved.segments.iter() {
            if !self.retained_mapping_is_current(&segment.mapping) {
                return Err(MaxwellGpuAccessError::StaleMapping {
                    mapping: segment.mapping.id,
                    generation: segment.mapping.generation,
                });
            }
        }
        Ok(())
    }
    /// Reports whether a retained mapping still names the active generation.
    #[must_use]
    pub fn retained_mapping_is_current(&self, retained: &MaxwellGpuMapping) -> bool {
        retained.owner.as_ptr() == Arc::as_ptr(&self.inner)
            && retained.current.load(Ordering::Acquire)
    }
    fn validate_resolution_request(
        &self,
        offset: GpuVirtualAddress,
    ) -> Result<(), MaxwellGpuAccessError> {
        if !self.initialized() {
            return Err(MaxwellGpuAccessError::NotInitialized);
        }
        self.address(offset.get())
            .map_err(MaxwellGpuAccessError::Address)?;
        Ok(())
    }
}
