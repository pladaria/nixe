//! Process/kernel mapping state. Guest bytes and native translations live in AddressSpace.
mod heap;
#[cfg(test)]
mod tests;
pub use heap::HeapResize;
use nixe_memory::{
    AddressSpace, AddressWidth, Allocation, AllocationFuture, AllocationRange, GuestVirtualAddress,
    MemoryAccount, MemoryError, MemoryPermissions, PAGE_SIZE,
};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKind {
    Data,
    Code,
    MutableCode,
    ModuleCode,
    MutableModuleCode,
    ThreadLocal,
    Stack,
    Heap,
    Shared,
}
impl MemoryKind {
    fn mutable(self) -> bool {
        matches!(
            self,
            Self::MutableCode | Self::MutableModuleCode | Self::Heap
        )
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mapping {
    pub kind: MemoryKind,
    pub permissions: MemoryPermissions,
    pub uncached: bool,
    pub locked: bool,
    pub permission_locked: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryInfo {
    pub base: GuestVirtualAddress,
    pub bytes: u64,
    pub mapping: Option<Mapping>,
}
#[derive(Debug)]
pub enum MappingError {
    InvalidRange,
    AlreadyMapped,
    Unmapped,
    StateMismatch,
    PhysicalIdentityMismatch,
    PermissionLocked,
    InvalidPermission,
    Memory(Arc<MemoryError>),
}
impl fmt::Display for MappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange => f.write_str("invalid kernel virtual-memory range"),
            Self::AlreadyMapped => f.write_str("kernel mapping overlaps mapped memory"),
            Self::Unmapped => f.write_str("kernel operation crosses unmapped memory"),
            Self::StateMismatch => {
                f.write_str("kernel operation does not match current mapping state")
            }
            Self::PhysicalIdentityMismatch => {
                f.write_str("mapping names another physical allocation")
            }
            Self::PermissionLocked => {
                f.write_str("mapping permissions are locked by an owning resource")
            }
            Self::InvalidPermission => f.write_str("kernel mapping permissions are unsupported"),
            Self::Memory(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for MappingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Memory(error) = self {
            Some(error.as_ref())
        } else {
            None
        }
    }
}
impl From<MemoryError> for MappingError {
    fn from(error: MemoryError) -> Self {
        if matches!(error, MemoryError::AliasMismatch) {
            Self::PhysicalIdentityMismatch
        } else {
            Self::Memory(Arc::new(error))
        }
    }
}
#[derive(Clone, Copy)]
struct Region {
    end: u64,
    mapping: Mapping,
}

/// Concrete process page-table owner. The interval tree contains kernel state,
/// not pages, RAM mirrors, resource authority or CPU access/translation state.
/// Allocation futures are obtained before taking this mapping writer; HLE keeps
/// its owned continuation while waiting, without holding a process/kernel lock.
pub struct ProcessMemory {
    space: Arc<AddressSpace>,
    account: MemoryAccount,
    system: MemoryAccount,
    regions: Mutex<BTreeMap<u64, Region>>,
    closed: AtomicBool,
    heap_region: crate::ProcessVirtualRegion,
    heap_bytes: AtomicU64,
    heap_requests: Mutex<heap::HeapRequests>,
    system_reservation: u64,
}
impl ProcessMemory {
    pub fn new(
        account: MemoryAccount,
        system: MemoryAccount,
        width: AddressWidth,
        heap_region: crate::ProcessVirtualRegion,
        system_reservation: u64,
    ) -> Result<Self, MemoryError> {
        if account.machine().identity() != system.machine().identity() {
            return Err(MemoryError::Invalid(
                "process and system accounts belong to different machines",
            ));
        }
        if heap_region.size() == 0
            || !heap_region.base().is_aligned_to(PAGE_SIZE)
            || !heap_region.size().is_multiple_of(PAGE_SIZE)
            || heap_region
                .base()
                .get()
                .checked_add(heap_region.size())
                .is_none_or(|end| end > width.bytes())
            || account
                .limit()
                .checked_add(system_reservation)
                .is_none_or(|limit| limit > account.machine().arena().capacity())
        {
            return Err(MemoryError::Invalid(
                "invalid process heap or physical resource budget",
            ));
        }
        Ok(Self {
            space: AddressSpace::new(account.machine().clone(), width)?,
            account,
            system,
            regions: Mutex::new(BTreeMap::new()),
            closed: AtomicBool::new(false),
            heap_region,
            heap_bytes: AtomicU64::new(0),
            heap_requests: Mutex::new(heap::HeapRequests::default()),
            system_reservation,
        })
    }
    pub fn space(&self) -> &Arc<AddressSpace> {
        &self.space
    }
    pub fn account(&self) -> &MemoryAccount {
        &self.account
    }
    pub fn system_account(&self) -> &MemoryAccount {
        &self.system
    }
    pub fn map_zeroed(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        mapping: Mapping,
    ) -> Result<(), MappingError> {
        // Horizon maps thread-local pages from kernel page buffers, outside
        // the process's PhysicalMemoryMax resource limit.
        // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_thread_local_page.cpp
        let account = if mapping.kind == MemoryKind::ThreadLocal {
            &self.system
        } else {
            &self.account
        };
        let allocation = account.allocate(bytes)?;
        self.map(start, allocation, 0, bytes, mapping)
    }
    pub fn heap_size(&self) -> u64 {
        if self.closed.load(Ordering::Acquire) {
            0
        } else {
            self.heap_bytes.load(Ordering::Acquire)
        }
    }
    pub fn accounting(&self) -> crate::ProcessMemoryAccounting {
        crate::ProcessMemoryAccounting::new(
            self.account.limit(),
            self.account.used_bytes(),
            self.heap_size(),
            self.system_reservation,
        )
    }
    pub fn allocate(&self, bytes: u64) -> AllocationFuture {
        self.account.allocate_async(bytes)
    }
    pub fn close(&self) -> Result<(), MappingError> {
        self.closed.store(true, Ordering::Release);
        let wake = self.heap_requests.lock().unwrap().wakers();
        for waker in wake {
            waker.wake_by_ref();
        }
        let mut regions = self.regions.lock().unwrap();
        self.space.close()?;
        regions.clear();
        Ok(())
    }
    fn end(&self, start: GuestVirtualAddress, bytes: u64) -> Result<u64, MappingError> {
        start
            .get()
            .checked_add(bytes)
            .filter(|end| {
                bytes != 0
                    && start.is_aligned_to(PAGE_SIZE)
                    && bytes.is_multiple_of(PAGE_SIZE)
                    && *end <= self.space.width().bytes()
            })
            .ok_or(MappingError::InvalidRange)
    }
    fn permissions(mapping: Mapping) -> Result<(), MappingError> {
        if mapping.permissions.contains(MemoryPermissions::WRITE)
            && mapping.permissions.contains(MemoryPermissions::EXECUTE)
        {
            Err(MappingError::InvalidPermission)
        } else {
            Ok(())
        }
    }
    pub fn map(
        &self,
        start: GuestVirtualAddress,
        allocation: Allocation,
        offset: u64,
        bytes: u64,
        mapping: Mapping,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        Self::permissions(mapping)?;
        let mut regions = self.regions.lock().unwrap();
        Self::unmapped(&regions, start.get(), end)?;
        self.space
            .map(start, allocation, offset, bytes, mapping.permissions)?;
        Self::insert(&mut regions, start.get(), Region { end, mapping });
        Ok(())
    }
    /// Initializes private RAM before publishing its final guest permissions.
    /// This is used while constructing a process, before it can run or export
    /// the allocation to graphics or another process.
    pub fn map_initialized(
        &self,
        start: GuestVirtualAddress,
        bytes: &[u8],
        mapping: Mapping,
    ) -> Result<(), MappingError> {
        self.end(start, bytes.len() as u64)?;
        Self::permissions(mapping)?;
        let allocation = self.account.allocate(bytes.len() as u64)?;
        let range = AllocationRange::new(allocation.pin(), 0, bytes.len() as u64)?;
        // The new allocation has no external observer or device producer.
        self.account.machine().write_resolved(&[range], bytes)?;
        self.map(start, allocation, 0, bytes.len() as u64, mapping)
    }
    pub fn unmap(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        expected: Mapping,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping == expected {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        self.space.unmap(start, bytes)?;
        Self::remove(&mut regions, start.get(), end);
        Ok(())
    }
    pub fn unmap_shared(
        &self,
        start: GuestVirtualAddress,
        allocation: &Allocation,
        bytes: u64,
        expected: Mapping,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping == expected {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        let captured = self.space.capture(start, bytes, MemoryPermissions::NONE)?;
        let expected_allocation = allocation.pin();
        let mut offset = 0;
        for range in &captured {
            if range.allocation().machine().identity() != expected_allocation.machine().identity()
                || range.allocation().identity() != allocation.identity()
                || range.offset() != offset
            {
                return Err(MappingError::PhysicalIdentityMismatch);
            }
            offset += range.bytes();
        }
        self.space.unmap(start, bytes)?;
        Self::remove(&mut regions, start.get(), end);
        drop(regions);
        drop(captured);
        Ok(())
    }
    pub(crate) fn lock_transfer(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(Mapping, nixe_memory::CapturedMemory), MappingError> {
        let end = self.end(start, bytes)?;
        if !matches!(
            permissions,
            MemoryPermissions::NONE | MemoryPermissions::READ | MemoryPermissions::READ_WRITE
        ) {
            return Err(MappingError::InvalidPermission);
        }
        let mut regions = self.regions.lock().unwrap();
        let before = regions
            .range(..=start.get())
            .next_back()
            .filter(|(_, region)| region.end > start.get())
            .ok_or(MappingError::Unmapped)?
            .1
            .mapping;
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping == before
                && mapping.kind.mutable()
                && !mapping.uncached
                && !mapping.permission_locked
                && !mapping.locked
                && mapping.permissions == MemoryPermissions::READ_WRITE
            {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        let ranges = self
            .space
            .capture(start, bytes, MemoryPermissions::READ_WRITE)?;
        let ranges = nixe_memory::CapturedMemory::new(ranges)?;
        self.space.protect(start, bytes, permissions)?;
        Self::update(&mut regions, start.get(), end, |mut mapping| {
            mapping.permissions = permissions;
            mapping.locked = true;
            mapping
        });
        Ok((before, ranges))
    }
    pub(crate) fn unlock_transfer(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
        before: Mapping,
        ranges: &nixe_memory::CapturedMemory,
    ) -> Result<(), MappingError> {
        let mut regions = self.regions.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let end = self.end(start, bytes)?;
        let locked = Mapping {
            permissions,
            locked: true,
            ..before
        };
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping == locked {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        let current = self.space.capture(start, bytes, MemoryPermissions::NONE)?;
        // Locked kernel mappings cannot be remapped; verify that invariant
        // before restoring access to the original owner.
        let (mut left, mut right, mut a, mut b) = (0, 0, 0, 0);
        while left < current.len() && right < ranges.len() {
            let first = &current[left];
            let second = &ranges[right];
            if first.allocation().identity() != second.allocation().identity()
                || first.offset() + a != second.offset() + b
            {
                return Err(MappingError::StateMismatch);
            }
            let count = (first.bytes() - a).min(second.bytes() - b);
            a += count;
            b += count;
            if a == first.bytes() {
                left += 1;
                a = 0;
            }
            if b == second.bytes() {
                right += 1;
                b = 0;
            }
        }
        if left != current.len() || right != ranges.len() {
            return Err(MappingError::StateMismatch);
        }
        self.space.protect(start, bytes, before.permissions)?;
        Self::update(&mut regions, start.get(), end, |_| before);
        Ok(())
    }
    /// The kernel validates both semantic states before the physical memory
    /// owner changes source permissions and destination mappings together.
    pub fn map_alias(
        &self,
        source: GuestVirtualAddress,
        destination: GuestVirtualAddress,
        bytes: u64,
        before: Mapping,
        after: Mapping,
        alias: Mapping,
    ) -> Result<(), MappingError> {
        let end = self.end(source, bytes)?;
        let destination_end = self.end(destination, bytes)?;
        Self::permissions(after)?;
        Self::permissions(alias)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, source.get(), end, |mapping| {
            if mapping == before {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        Self::unmapped(&regions, destination.get(), destination_end)?;
        self.space.map_alias(
            source,
            destination,
            bytes,
            after.permissions,
            alias.permissions,
        )?;
        Self::remove(&mut regions, source.get(), end);
        Self::insert(
            &mut regions,
            source.get(),
            Region {
                end,
                mapping: after,
            },
        );
        Self::insert(
            &mut regions,
            destination.get(),
            Region {
                end: destination_end,
                mapping: alias,
            },
        );
        Ok(())
    }
    pub fn unmap_alias(
        &self,
        source: GuestVirtualAddress,
        destination: GuestVirtualAddress,
        bytes: u64,
        before: Mapping,
        restored: Mapping,
        alias: Mapping,
    ) -> Result<(), MappingError> {
        let end = self.end(source, bytes)?;
        let destination_end = self.end(destination, bytes)?;
        Self::permissions(restored)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, source.get(), end, |mapping| {
            if mapping == before {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        Self::covered(&regions, destination.get(), destination_end, |mapping| {
            if mapping == alias {
                Ok(())
            } else {
                Err(MappingError::StateMismatch)
            }
        })?;
        self.space
            .unmap_alias(source, destination, bytes, restored.permissions)?;
        Self::remove(&mut regions, destination.get(), destination_end);
        Self::remove(&mut regions, source.get(), end);
        Self::insert(
            &mut regions,
            source.get(),
            Region {
                end,
                mapping: restored,
            },
        );
        Ok(())
    }
    pub fn protect(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        if permissions.contains(MemoryPermissions::WRITE)
            && permissions.contains(MemoryPermissions::EXECUTE)
        {
            return Err(MappingError::InvalidPermission);
        }
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping.locked || mapping.permission_locked {
                Err(MappingError::PermissionLocked)
            } else if mapping.uncached || !mapping.kind.mutable() {
                Err(MappingError::StateMismatch)
            } else {
                Ok(())
            }
        })?;
        self.space.protect(start, bytes, permissions)?;
        Self::update(&mut regions, start.get(), end, |mut mapping| {
            mapping.permissions = permissions;
            mapping
        });
        Ok(())
    }
    pub fn set_uncached(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        uncached: bool,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping.locked {
                Err(MappingError::PermissionLocked)
            } else if !mapping.kind.mutable() {
                Err(MappingError::StateMismatch)
            } else {
                Ok(())
            }
        })?;
        Self::update(&mut regions, start.get(), end, |mut mapping| {
            mapping.uncached = uncached;
            mapping
        });
        Ok(())
    }
    pub fn lock_permissions(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let mut regions = self.regions.lock().unwrap();
        Self::covered(&regions, start.get(), end, |mapping| {
            if mapping.locked {
                Err(MappingError::PermissionLocked)
            } else if !matches!(
                mapping.kind,
                MemoryKind::MutableCode | MemoryKind::MutableModuleCode
            ) {
                Err(MappingError::StateMismatch)
            } else {
                Ok(())
            }
        })?;
        Self::update(&mut regions, start.get(), end, |mut mapping| {
            mapping.permission_locked = true;
            mapping
        });
        Ok(())
    }
    pub fn query(&self, address: GuestVirtualAddress) -> Result<MemoryInfo, MappingError> {
        let address = address.get();
        if address >= self.space.width().bytes() {
            return Err(MappingError::InvalidRange);
        }
        let regions = self.regions.lock().unwrap();
        let before = regions.range(..=address).next_back();
        if let Some((&base, region)) = before.filter(|(_, region)| region.end > address) {
            return Ok(MemoryInfo {
                base: GuestVirtualAddress::new(base),
                bytes: region.end - base,
                mapping: Some(region.mapping),
            });
        }
        let base = before.map_or(0, |(_, region)| region.end);
        let end = regions
            .range(address..)
            .next()
            .map_or(self.space.width().bytes(), |(&base, _)| base);
        Ok(MemoryInfo {
            base: GuestVirtualAddress::new(base),
            bytes: end - base,
            mapping: None,
        })
    }
    fn unmapped(regions: &BTreeMap<u64, Region>, start: u64, end: u64) -> Result<(), MappingError> {
        if regions
            .range(..end)
            .next_back()
            .is_some_and(|(_, region)| region.end > start)
        {
            Err(MappingError::AlreadyMapped)
        } else {
            Ok(())
        }
    }
    fn covered(
        regions: &BTreeMap<u64, Region>,
        mut start: u64,
        end: u64,
        mut validate: impl FnMut(Mapping) -> Result<(), MappingError>,
    ) -> Result<(), MappingError> {
        while start < end {
            let (_, region) = regions
                .range(..=start)
                .next_back()
                .filter(|(_, region)| region.end > start)
                .ok_or(MappingError::Unmapped)?;
            validate(region.mapping)?;
            start = region.end.min(end);
        }
        Ok(())
    }
    fn split(regions: &mut BTreeMap<u64, Region>, address: u64) {
        if let Some((&base, &region)) = regions
            .range(..address)
            .next_back()
            .filter(|(_, region)| region.end > address)
        {
            regions.insert(
                base,
                Region {
                    end: address,
                    mapping: region.mapping,
                },
            );
            regions.insert(address, region);
        }
    }
    fn remove(regions: &mut BTreeMap<u64, Region>, start: u64, end: u64) {
        Self::split(regions, start);
        Self::split(regions, end);
        let keys: Vec<_> = regions.range(start..end).map(|(&base, _)| base).collect();
        for key in keys {
            regions.remove(&key);
        }
    }
    fn insert(regions: &mut BTreeMap<u64, Region>, mut start: u64, mut region: Region) {
        if let Some((&base, _)) = regions
            .range(..start)
            .next_back()
            .filter(|(_, before)| before.end == start && before.mapping == region.mapping)
        {
            regions.remove(&base);
            start = base;
        }
        if let Some(after) = regions
            .get(&region.end)
            .copied()
            .filter(|after| after.mapping == region.mapping)
        {
            regions.remove(&region.end);
            region.end = after.end;
        }
        regions.insert(start, region);
    }
    fn update(
        regions: &mut BTreeMap<u64, Region>,
        start: u64,
        end: u64,
        mut update: impl FnMut(Mapping) -> Mapping,
    ) {
        Self::split(regions, start);
        Self::split(regions, end);
        let changed: Vec<_> = regions
            .range(start..end)
            .map(|(&base, region)| {
                (
                    base,
                    Region {
                        end: region.end,
                        mapping: update(region.mapping),
                    },
                )
            })
            .collect();
        for (base, _) in &changed {
            regions.remove(base);
        }
        for (base, region) in changed {
            Self::insert(regions, base, region);
        }
    }
}
