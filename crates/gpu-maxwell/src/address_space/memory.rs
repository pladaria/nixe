//! The shared GPU page table used directly by the command processor.
use super::*;
use std::sync::{RwLock, RwLockReadGuard, atomic::AtomicU64};

#[derive(Debug)]
pub(super) struct TranslationState {
    id: MaxwellAddressSpaceId,
    machine: Machine,
    profile: MaxwellGpuProfile,
    pub(super) mappings: RwLock<Mappings>,
    pub(super) generation: AtomicU64,
    pub(super) initialized: AtomicBool,
}
#[derive(Debug)]
pub(super) struct Mappings {
    pub(super) table: PageTable,
    pub(super) live: Vec<Option<LiveMapping>>,
}
/// Cloning a channel's memory handle shares translation metadata. It neither
/// clones the address-space allocator nor snapshots its mapping tree/RAM.
#[derive(Clone, Debug)]
pub struct MaxwellGpuMemory {
    pub(super) inner: Arc<TranslationState>,
}
pub(crate) struct MaxwellGpuReadBatch<'a> {
    space: &'a MaxwellGpuMemory,
    mappings: Option<RwLockReadGuard<'a, Mappings>>,
    reader: nixe_memory::MappedRamReader<'a>,
}

impl MaxwellGpuReadBatch<'_> {
    pub(crate) fn release(&mut self) {
        self.reader.release();
        self.mappings = None;
    }

    pub(crate) fn read_into(
        &mut self,
        address: GpuVirtualAddress,
        output: &mut [u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let space = self.space;
        space.validate_resolution_request(address, MemoryPermissions::READ)?;
        if output.is_empty() {
            return Ok(());
        }
        space
            .checked_add(address, output.len() as u64 - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        if self.mappings.is_none() {
            self.mappings = Some(
                space
                    .inner
                    .mappings
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }
        let mappings = self.mappings.as_ref().unwrap();
        let mut copied = 0;
        while copied < output.len() {
            let cursor = space
                .checked_add(address, copied as u64)
                .map_err(MaxwellGpuAccessError::Address)?;
            let entry = mappings
                .table
                .get(cursor.get())
                .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: cursor })?;
            let mapping = &mappings.live[entry.slot as usize]
                .as_ref()
                .unwrap()
                .descriptor;
            if !mapping.permissions.contains(MemoryPermissions::READ) {
                return Err(MaxwellGpuAccessError::PermissionDenied {
                    address: cursor,
                    required: MemoryPermissions::READ,
                    available: mapping.permissions,
                });
            }
            let within = cursor.get() & 0xfff;
            let bytes = (0x1000 - within).min((output.len() - copied) as u64) as usize;
            let physical = ArenaOffset::new((u64::from(entry.physical_page) << 12) + within);
            // SAFETY: the mapping guard retains the live physical pins.
            if !unsafe {
                self.reader
                    .try_read(physical, &mut output[copied..copied + bytes])
            }
            .map_err(|error| MaxwellGpuAccessError::Backing(Arc::new(error)))?
            {
                let resolved = space.resolve_range_locked(
                    address,
                    output.len() as u64,
                    MemoryPermissions::READ,
                    mappings,
                )?;
                let inputs = space.capture_validated(&resolved)?;
                self.release();
                return MaxwellGpuMemory::read_device(inputs, output, access);
            }
            copied += bytes;
        }
        Ok(())
    }
}
impl MaxwellGpuMemory {
    pub(crate) fn read_batch(&self) -> MaxwellGpuReadBatch<'_> {
        MaxwellGpuReadBatch {
            space: self,
            mappings: None,
            reader: self.machine().mapped_ram_reader(),
        }
    }
    pub(super) fn new(
        machine: Machine,
        id: MaxwellAddressSpaceId,
        profile: MaxwellGpuProfile,
    ) -> Self {
        Self {
            inner: Arc::new(TranslationState {
                id,
                machine,
                profile,
                mappings: RwLock::new(Mappings {
                    table: PageTable::new(profile.virtual_address().address_bits().bits()),
                    live: vec![None],
                }),
                generation: AtomicU64::new(MappingGeneration::INITIAL.get()),
                initialized: AtomicBool::new(false),
            }),
        }
    }
    pub fn machine(&self) -> &Machine {
        &self.inner.machine
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

    /// Translate under the same mapping lifetime boundary used by immediate reads.
    #[inline]
    pub fn translate(
        &self,
        offset: GpuVirtualAddress,
        permissions: MemoryPermissions,
    ) -> Result<MaxwellGpuTranslation, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset, permissions)?;
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = mappings
            .table
            .get(offset.get())
            .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: offset })?;
        let mapping = &mappings.live[entry.slot as usize]
            .as_ref()
            .unwrap()
            .descriptor;
        if !mapping.permissions.contains(permissions) {
            return Err(MaxwellGpuAccessError::PermissionDenied {
                address: offset,
                required: permissions,
                available: mapping.permissions,
            });
        }
        let within = offset.get() & 0xfff;
        Ok(MaxwellGpuTranslation {
            allocation: entry.allocation,
            physical: ArenaOffset::new((u64::from(entry.physical_page) << 12) + within),
            bytes: 0x1000 - within,
            generation: mapping.generation,
            permissions: mapping.permissions,
            kind: mapping.kind,
        })
    }
    /// Resolves the maximal active mapping suffix beginning at `offset`.
    pub fn resolve_maximal(
        &self,
        offset: GpuVirtualAddress,
        permissions: MemoryPermissions,
    ) -> Result<MaxwellResolvedMapping, MaxwellGpuAccessError> {
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.resolve_maximal_locked(offset, permissions, &mappings)
    }

    fn resolve_maximal_locked(
        &self,
        offset: GpuVirtualAddress,
        permissions: MemoryPermissions,
        mappings: &Mappings,
    ) -> Result<MaxwellResolvedMapping, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset, permissions)?;
        let mapping = mappings
            .table
            .get(offset.get())
            .and_then(|entry| mappings.live[entry.slot as usize].as_ref())
            .map(|live| &live.descriptor)
            .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: offset })?;
        if !mapping.permissions.contains(permissions) {
            return Err(MaxwellGpuAccessError::PermissionDenied {
                address: offset,
                required: permissions,
                available: mapping.permissions,
            });
        }
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
        permissions: MemoryPermissions,
    ) -> Result<MaxwellResolvedRange, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset, permissions)?;
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
        self.resolve_range_locked(offset, size, permissions, &mappings)
    }
    fn resolve_range_locked(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
        mappings: &Mappings,
    ) -> Result<MaxwellResolvedRange, MaxwellGpuAccessError> {
        let mut remaining = size;
        let mut cursor = offset;
        let mut segments = Vec::new();
        while remaining != 0 {
            let mut segment = self.resolve_maximal_locked(cursor, permissions, mappings)?;
            segment.size = segment.size.min(remaining);
            remaining -= segment.size;
            if remaining != 0 {
                cursor = self
                    .checked_add(cursor, segment.size)
                    .map_err(MaxwellGpuAccessError::Address)?;
            }
            segments.push(segment);
        }
        Ok(MaxwellResolvedRange {
            address_space: self.inner.id,
            offset,
            size,
            permissions,
            segments: segments.into(),
        })
    }
    /// Consume an owner write directly through the page table, as Eden's
    /// WriteBlockCached does. Only overlapping command RAM needs a temporary copy.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
    pub(crate) fn write_from(
        &self,
        address: GpuVirtualAddress,
        source: &[u8],
        written: &mut impl FnMut(
            &[(nixe_memory::AllocationId, nixe_memory::Extent)],
        ) -> Result<(), Arc<MemoryError>>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.write_upload(address, nixe_memory::WriteSource::Bytes(source), written)
    }
    pub(crate) fn write_words(
        &self,
        address: GpuVirtualAddress,
        source: crate::MaxwellMethodSpan<'_>,
        written: &mut impl FnMut(
            &[(nixe_memory::AllocationId, nixe_memory::Extent)],
        ) -> Result<(), Arc<MemoryError>>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let source = if let Some(bytes) = source.gathered_bytes() {
            nixe_memory::WriteSource::Bytes(bytes)
        } else {
            let (address, offset) = source.ram_bytes().expect("command source");
            nixe_memory::WriteSource::Ram {
                address,
                offset,
                bytes: source.len() * 4,
            }
        };
        self.write_upload(address, source, written)
    }
    fn write_upload(
        &self,
        address: GpuVirtualAddress,
        source: nixe_memory::WriteSource<'_>,
        written: &mut impl FnMut(
            &[(nixe_memory::AllocationId, nixe_memory::Extent)],
        ) -> Result<(), Arc<MemoryError>>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.validate_resolution_request(address, MemoryPermissions::WRITE)?;
        if source.is_empty() {
            return Ok(());
        }
        self.checked_add(address, source.len() as u64 - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut extents: smallvec::SmallVec<[(nixe_memory::AllocationId, nixe_memory::Extent); 4]> =
            smallvec::SmallVec::new();
        let mut offset = 0;
        while offset < source.len() as u64 {
            let cursor = self
                .checked_add(address, offset)
                .map_err(MaxwellGpuAccessError::Address)?;
            let entry = mappings
                .table
                .get(cursor.get())
                .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: cursor })?;
            let mapping = &mappings.live[entry.slot as usize]
                .as_ref()
                .unwrap()
                .descriptor;
            if !mapping.permissions.contains(MemoryPermissions::WRITE) {
                return Err(MaxwellGpuAccessError::PermissionDenied {
                    address: cursor,
                    required: MemoryPermissions::WRITE,
                    available: mapping.permissions,
                });
            }
            let within = cursor.get() & 0xfff;
            let bytes = (0x1000 - within).min(source.len() as u64 - offset);
            let start = ArenaOffset::new((u64::from(entry.physical_page) << 12) + within);
            if let Some((allocation, last)) = extents.last_mut()
                && *allocation == entry.allocation
                && last.start.get() + last.bytes == start.get()
            {
                last.bytes += bytes;
            } else {
                extents.push((entry.allocation, nixe_memory::Extent { start, bytes }));
            }
            offset += bytes;
        }
        // Keep the mapping guard through the write and exact authority update.
        // No retained request or second address translation is needed on the owner.
        let device = unsafe { self.machine().write_mapped_owner(&extents, source) }
            .map_err(|error| MaxwellGpuAccessError::Backing(Arc::new(error)))?;
        if device {
            written(&extents).map_err(MaxwellGpuAccessError::Backing)?;
        }
        Ok(())
    }
    /// Consume immediate RAM under one mapping lifetime boundary. Retained
    /// resolution objects are only needed when device coherence can suspend.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
    pub fn read_into(
        &self,
        address: GpuVirtualAddress,
        output: &mut [u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        self.read_batch().read_into(address, output, access)
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
                    .physical
                    .pin(segment.backing_offset, segment.size)
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
    /// Eden GuestMemory::TrySetSpan consumes contiguous command RAM directly.
    /// Retain the exact source instead of holding a mapping lock while engine
    /// effects may suspend or service concurrent remaps.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/guest_memory.h
    pub(crate) fn command_ram(
        &self,
        resolved: &MaxwellResolvedRange,
    ) -> Result<Option<(nixe_memory::HostAddress, AllocationRange)>, MaxwellGpuAccessError> {
        let _change = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.validate_resolved(resolved)?;
        let [segment] = resolved.segments() else {
            return Ok(None);
        };
        self.command_ram_segment(segment)
    }
    /// The common single-mapping command entry needs no scatter list allocation.
    pub(crate) fn command_ram_direct(
        &self,
        address: GpuVirtualAddress,
        size: u64,
    ) -> Result<
        Option<(
            MaxwellResolvedMapping,
            nixe_memory::HostAddress,
            AllocationRange,
        )>,
        MaxwellGpuAccessError,
    > {
        self.validate_resolution_request(address, MemoryPermissions::READ)?;
        if size == 0 {
            return Err(MaxwellGpuAccessError::EmptyRange);
        }
        self.checked_add(address, size - 1)
            .map_err(MaxwellGpuAccessError::Address)?;
        let mappings = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut segment =
            self.resolve_maximal_locked(address, MemoryPermissions::READ, &mappings)?;
        if segment.size < size {
            return Ok(None);
        }
        segment.size = size;
        Ok(self
            .command_ram_segment(&segment)?
            .map(|(pointer, pin)| (segment, pointer, pin)))
    }
    fn command_ram_segment(
        &self,
        segment: &MaxwellResolvedMapping,
    ) -> Result<Option<(nixe_memory::HostAddress, AllocationRange)>, MaxwellGpuAccessError> {
        let mapping = segment.mapping().physical();
        let begin = segment.backing_offset();
        let spans = mapping.spans();
        let index = spans.partition_point(|span| span.logical + span.bytes <= begin);
        let Some(span) = spans.get(index).filter(|span| {
            span.logical <= begin && begin + segment.size() <= span.logical + span.bytes
        }) else {
            return Ok(None);
        };
        let start = ArenaOffset::new(span.start.get() + begin - span.logical);
        let backing = |error| MaxwellGpuAccessError::Backing(Arc::new(error));
        let range = mapping
            .pin_span(index, start, segment.size())
            .map_err(backing)?
            .ok_or_else(|| backing(MemoryError::Invalid("command RAM has retired")))?;
        Ok(self
            .machine()
            .try_ram_pointer(&range)
            .map_err(backing)?
            .map(|pointer| (pointer, range)))
    }
    /// Read accepted spans. Device-dependent results own their bytes; a pending
    /// access never retains this output buffer or a borrowed frontend stack.
    pub fn read_resolved(
        &self,
        resolved: &MaxwellResolvedRange,
        output: &mut [u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        if !resolved.permissions.contains(MemoryPermissions::READ) {
            return Err(MaxwellGpuAccessError::PermissionDenied {
                address: resolved.offset,
                required: MemoryPermissions::READ,
                available: resolved.permissions,
            });
        }
        if output.len() as u64 != resolved.size {
            return Err(MaxwellGpuAccessError::OutputSizeMismatch {
                expected: resolved.size,
                actual: output.len() as u64,
            });
        }
        let change = self
            .inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.validate_resolved(resolved)?;
        let mut copied = 0;
        // ReadBlock consumes directly translated physical runs, as in Eden:
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
        // Live GPU mappings own their physical ranges while the mapping lock is
        // held. Ordinary reads need neither another pin vector nor boxed bytes.
        for segment in resolved.segments.iter() {
            let begin = segment.backing_offset;
            let end = begin + segment.size;
            let spans = segment.mapping.physical.spans();
            let first = spans.partition_point(|span| span.logical + span.bytes <= begin);
            for span in &spans[first..] {
                if span.logical >= end {
                    break;
                }
                let low = begin.max(span.logical);
                let high = end.min(span.logical + span.bytes);
                let bytes = (high - low) as usize;
                let physical = ArenaOffset::new(span.start.get() + low - span.logical);
                // SAFETY: validation and the mapping read lock retain the
                // original live physical ranges through this raw RAM copy.
                if !unsafe {
                    self.machine()
                        .try_read_mapped_ram(physical, &mut output[copied..copied + bytes])
                }
                .map_err(|error| MaxwellGpuAccessError::Backing(Arc::new(error)))?
                {
                    // Only the device-dependent operation needs owned inputs.
                    // Pin before dropping mapping exclusion; the owner may
                    // wait, unmap or retire mappings while resolving its bytes.
                    let ranges = self.capture_validated(resolved)?;
                    drop(change);
                    return Self::read_device(ranges, output, access);
                }
                copied += bytes;
            }
        }
        debug_assert_eq!(copied, output.len());
        Ok(())
    }
    fn read_device(
        ranges: Vec<AllocationRange>,
        output: &mut [u8],
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), MaxwellGpuAccessError> {
        let nixe_memory::AccessResult::Read(bytes) =
            access(ranges, nixe_memory::AccessOperation::Read)
                .map_err(MaxwellGpuAccessError::Backing)?
        else {
            unreachable!("read returns bytes")
        };
        if bytes.len() != output.len() {
            return Err(MaxwellGpuAccessError::OutputSizeMismatch {
                expected: output.len() as u64,
                actual: bytes.len() as u64,
            });
        }
        output.copy_from_slice(&bytes);
        Ok(())
    }
    pub fn translation_metadata_bytes(&self) -> usize {
        self.inner
            .mappings
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .table
            .metadata_bytes()
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
        permissions: MemoryPermissions,
    ) -> Result<(), MaxwellGpuAccessError> {
        if !self.initialized() {
            return Err(MaxwellGpuAccessError::NotInitialized);
        }
        self.address(offset.get())
            .map_err(MaxwellGpuAccessError::Address)?;
        if permissions == MemoryPermissions::NONE {
            return Err(MaxwellGpuAccessError::InvalidPermissions);
        }
        Ok(())
    }
}
