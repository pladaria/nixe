//! Physical device coverage and ordered CPU/device RAM access.
use crate::{AddressSpace, AllocationRange, ArenaOffset, Extent, Machine, MemoryError, PAGE_SIZE};
use std::sync::{
    Arc, RwLock, Weak,
    atomic::{AtomicU64, Ordering},
};

pub(crate) struct Tracking {
    device_memory: RwLock<Option<Weak<crate::device_memory::Inner>>>,
    protection_calls: AtomicU64,
    #[cfg(any(test, feature = "performance-counters"))]
    captured_bytes: AtomicU64,
}
/// The two sources consumed synchronously by graphics-owner RAM writes.
#[derive(Clone, Copy)]
pub enum WriteSource<'a> {
    Bytes(&'a [u8]),
    Ram {
        address: crate::HostAddress,
        offset: usize,
        bytes: usize,
    },
}
impl<'a> WriteSource<'a> {
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Ram { bytes, .. } => *bytes,
        }
    }
    pub fn slice(self, range: std::ops::Range<usize>) -> WriteSource<'a> {
        assert!(range.start <= range.end && range.end <= self.len());
        match self {
            Self::Bytes(bytes) => Self::Bytes(&bytes[range]),
            Self::Ram {
                address, offset, ..
            } => Self::Ram {
                address,
                offset: offset.checked_add(range.start).expect("RAM source offset"),
                bytes: range.len(),
            },
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
/// A graphics-owner capture retains and admits these physical regions until the
/// callback returns. RAM is read through instruction primitives, never borrowed
/// as a Rust slice; other guest accesses can use mixed widths concurrently.
pub struct ResourceReader<'a> {
    machine: &'a Machine,
    retained: Vec<Extent>,
    #[cfg(any(test, feature = "performance-counters"))]
    captured: std::cell::Cell<u64>,
}
/// Admitted GPU publication into retained RAM, without exposing Rust references
/// to memory concurrently accessed by native guest instructions.
pub struct ResourceWriter<'a> {
    machine: &'a Machine,
    retained: Vec<Extent>,
}
impl ResourceWriter<'_> {
    pub fn write(&self, extent: Extent, source: &[u8]) -> Result<(), MemoryError> {
        let index = self
            .retained
            .partition_point(|retained| retained.start <= extent.start);
        if extent.bytes != source.len() as u64
            || index == 0
            || extent
                .start
                .get()
                .checked_add(extent.bytes)
                .is_none_or(|end| {
                    end > self.retained[index - 1].start.get() + self.retained[index - 1].bytes
                })
        {
            return Err(MemoryError::Invalid("GPU publication exceeds retained RAM"));
        }
        // SAFETY: write_resources retains these pages and excludes physical
        // access for the entire callback; the source is owned host memory.
        unsafe {
            self.machine.write_admitted(extent, source);
        }
        Ok(())
    }
}

/// A validated region within the current capture. It cannot outlive admission.
pub struct ResourceRead<'a> {
    source: crate::HostAddress,
    bytes: u64,
    _capture: std::marker::PhantomData<&'a ()>,
    #[cfg(any(test, feature = "performance-counters"))]
    captured: &'a std::cell::Cell<u64>,
}
impl ResourceReader<'_> {
    fn new<'a>(
        machine: &'a Machine,
        inputs: &[AllocationRange],
    ) -> Result<ResourceReader<'a>, MemoryError> {
        machine.validate_ranges(inputs)?;
        let mut extents: Vec<_> = inputs
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .collect();
        extents.sort_unstable_by_key(|extent| extent.start);
        let mut retained: Vec<Extent> = Vec::new();
        for extent in extents {
            if let Some(last) = retained.last_mut()
                && last.start.get() + last.bytes >= extent.start.get()
            {
                last.bytes = (extent.start.get() + extent.bytes).max(last.start.get() + last.bytes)
                    - last.start.get();
            } else {
                retained.push(extent);
            }
        }
        Ok(ResourceReader {
            machine,
            retained,
            #[cfg(any(test, feature = "performance-counters"))]
            captured: std::cell::Cell::new(0),
        })
    }
    pub fn machine(&self) -> crate::MachineId {
        self.machine.identity()
    }
    pub fn regions(&self) -> &[Extent] {
        &self.retained
    }
    fn validate(&self, extent: Extent) -> Result<(), MemoryError> {
        let index = self
            .retained
            .partition_point(|retained| retained.start <= extent.start);
        if extent.bytes == 0
            || index == 0
            || extent
                .start
                .get()
                .checked_add(extent.bytes)
                .is_none_or(|end| {
                    end > self.retained[index - 1].start.get() + self.retained[index - 1].bytes
                })
        {
            return Err(MemoryError::Invalid(
                "physical capture exceeds retained input",
            ));
        }
        Ok(())
    }
    pub fn region(&self, extent: Extent) -> Result<ResourceRead<'_>, MemoryError> {
        self.validate(extent)?;
        Ok(self.validated_region(extent))
    }
    fn validated_region(&self, extent: Extent) -> ResourceRead<'_> {
        ResourceRead {
            source: self
                .machine
                .arena()
                .host_address(extent.start, extent.bytes)
                .expect("retained physical region"),
            bytes: extent.bytes,
            _capture: std::marker::PhantomData,
            #[cfg(any(test, feature = "performance-counters"))]
            captured: &self.captured,
        }
    }
}
impl ResourceRead<'_> {
    #[inline]
    pub fn copy(&self, offset: u64, destination: &mut [u8]) -> Result<(), MemoryError> {
        if offset
            .checked_add(destination.len() as u64)
            .is_none_or(|end| end > self.bytes)
        {
            return Err(MemoryError::Invalid(
                "physical capture exceeds admitted region",
            ));
        }
        // SAFETY: the region is retained/admitted for this callback; the checked
        // destination is exclusive host memory. No guest RAM references escape.
        unsafe {
            crate::raw::capture(self.source.as_ptr().add(offset as usize), destination);
        }
        #[cfg(any(test, feature = "performance-counters"))]
        self.captured
            .set(self.captured.get() + destination.len() as u64);
        Ok(())
    }
}
/// A checked copy from retained physical input into an owned native snapshot.
#[derive(Clone, Copy, Debug)]
pub struct PhysicalCapture {
    pub source: Extent,
    pub destination: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrackingStatistics {
    pub protection_calls: u64,
    pub captured_bytes: u64,
}
impl Tracking {
    pub fn new(_bytes: u64) -> Self {
        Self {
            device_memory: RwLock::new(None),
            protection_calls: AtomicU64::new(0),
            #[cfg(any(test, feature = "performance-counters"))]
            captured_bytes: AtomicU64::new(0),
        }
    }

    pub fn record_protection(&self) {
        self.protection_calls.fetch_add(1, Ordering::Relaxed);
    }
    // Caller excludes physical writers, or owns the last allocation pin before
    // recycling. Invalidation precedes both the store and direct write access.

    fn statistics(&self) -> TrackingStatistics {
        TrackingStatistics {
            protection_calls: self.protection_calls.load(Ordering::Relaxed),
            #[cfg(any(test, feature = "performance-counters"))]
            captured_bytes: self.captured_bytes.load(Ordering::Relaxed),
            #[cfg(not(any(test, feature = "performance-counters")))]
            captured_bytes: 0,
        }
    }
}
impl crate::NativeAdmission<'_> {
    /// OnCPUWrite is called after suspending native admission. The operation
    /// and its possible journal pages were captured before that suspension.
    /// Buffer GPU-dirty stores are collected without waiting for a GPU fence.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/memory.cpp#L695-L735
    pub fn access_captured(
        &self,
        ranges: &[AllocationRange],
        cached: bool,
        mut operation: crate::AccessOperation,
    ) -> Result<crate::Access<crate::AccessResult>, MemoryError> {
        assert!(
            !self.is_active(),
            "CPU cache notification requires suspended admission"
        );
        let machine = self.address_space().machine();
        let notify = matches!(operation, crate::AccessOperation::Atomic(_))
            || (matches!(operation, crate::AccessOperation::Write(_)) && cached);
        let collect = notify
            && machine.notify_cpu_write(ranges.iter().flat_map(AllocationRange::physical_extents));
        if collect
            && let crate::AccessOperation::Write(bytes) = &mut operation
            && self.collect_cached_write(ranges, bytes)?
        {
            return Ok(crate::Access::Ready(crate::AccessResult::Written(
                std::mem::take(bytes),
            )));
        }
        machine.finish_captured_access(ranges, operation)
    }
    fn collect_cached_write(
        &self,
        ranges: &[AllocationRange],
        source: &[u8],
    ) -> Result<bool, MemoryError> {
        let machine = self.address_space().machine();
        if machine.validate_ranges(ranges)? != source.len() {
            return Err(MemoryError::Invalid("cached CPU store size mismatch"));
        }
        if source.is_empty() || source.len() > 16 {
            return Ok(false);
        }
        let extents: smallvec::SmallVec<[_; 2]> = ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .collect();
        let written = self.collect_write(&extents, || {
            let mut offset = 0;
            for &extent in &extents {
                let end = offset + extent.bytes as usize;
                unsafe {
                    machine.write_admitted(extent, &source[offset..end]);
                }
                offset = end;
            }
        })?;
        if written
            && extents.len() == 1
            && extents[0].start.get() / PAGE_SIZE
                == (extents[0].start.get() + extents[0].bytes - 1) / PAGE_SIZE
        {
            self.cpu_writes()
                .last_gpu_page
                .store(extents[0].start.get() / PAGE_SIZE, Ordering::Release);
        }
        Ok(written)
    }
}

impl Machine {
    pub(crate) fn bind_device_memory(
        &self,
        device: &crate::device_memory::DeviceMemory,
    ) -> Result<(), MemoryError> {
        let mut selected = self.tracking().device_memory.write().unwrap();
        if let Some(previous) = selected.as_ref().and_then(Weak::upgrade)
            && !Arc::ptr_eq(&previous, &device.0)
        {
            return Err(MemoryError::Invalid(
                "RAM already has a live GPU device memory manager",
            ));
        }
        *selected = Some(Arc::downgrade(&device.0));
        Ok(())
    }
    pub(crate) fn gpu_device_memory(&self) -> Option<crate::device_memory::DeviceMemory> {
        self.tracking()
            .device_memory
            .read()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade)
            .map(crate::device_memory::DeviceMemory)
    }
    pub(crate) fn refresh_cached_cpu_region(
        &self,
        source: &AddressSpace,
        start: crate::GuestVirtualAddress,
        bytes: u64,
        device: &crate::device_memory::DeviceMemory,
    ) -> Result<(), MemoryError> {
        let mut spaces = Vec::new();
        let aliases = self.aliases();
        let changed = source.cached_physical_regions(start, bytes)?;
        for (physical, bytes) in changed.iter() {
            let mut selected = Vec::new();
            aliases
                .index
                .overlapping(physical, physical + bytes, |alias| {
                    if let Some(space) = alias.space.upgrade() {
                        let begin = physical.max(alias.physical);
                        let end = (physical + bytes).min(alias.physical + alias.bytes);
                        selected.push((
                            space,
                            alias.virtual_address + begin - alias.physical,
                            end - begin,
                        ));
                    }
                });
            for (space, address, bytes) in selected {
                space.publish_cached_region(
                    crate::GuestVirtualAddress::new(address),
                    bytes,
                    device,
                )?;
                spaces.push(space);
            }
        }
        drop(aliases);
        drop(spaces);
        Ok(())
    }
    pub(crate) fn must_flush_physical(&self, extent: Extent) -> bool {
        let Some(device) = self.gpu_device_memory() else {
            return false;
        };
        let Some(interface) = device.interface() else {
            return false;
        };
        let mut flush = false;
        let scratch = std::cell::RefCell::new(Vec::new());
        device.apply_on_physical(extent, &scratch, |address, bytes| {
            flush |= interface.must_flush_region(address, bytes)
        });
        flush
    }
    pub(crate) fn notify_cpu_write(&self, extents: impl Iterator<Item = Extent>) -> bool {
        let Some(device) = self.gpu_device_memory() else {
            return false;
        };
        let Some(rasterizer) = device.interface() else {
            return false;
        };
        let mut collect = false;
        let scratch = std::cell::RefCell::new(Vec::new());
        for extent in extents {
            device.apply_on_physical(extent, &scratch, |address, bytes| {
                collect |= rasterizer.on_cpu_write(address, bytes);
            });
        }
        collect
    }
}

impl Machine {
    /// # Safety
    /// Caller retains/admitted this exact physical span and has resolved its
    /// device/cache dependencies. Ordinary stores need no reservation lookup;
    /// exclusive/RMW commit uses host CAS against these same canonical bytes.
    pub(crate) unsafe fn write_admitted(&self, extent: Extent, source: &[u8]) {
        debug_assert_eq!(extent.bytes as usize, source.len());
        let destination = self
            .arena()
            .host_address(extent.start, extent.bytes)
            .expect("retained physical store");
        unsafe {
            crate::raw::write(destination.as_ptr(), source);
        }
    }
    pub fn tracking_statistics(&self) -> TrackingStatistics {
        self.tracking().statistics()
    }
    fn atomic_extent(&self, range: &AllocationRange) -> Result<Extent, MemoryError> {
        self.validate_ranges(std::slice::from_ref(range))?;
        if !matches!(range.bytes(), 1 | 2 | 4 | 8 | 16) {
            return Err(MemoryError::Invalid("unsupported atomic access width"));
        }
        let mut extents = range.physical_extents();
        let extent = extents.next().expect("nonempty retained range");
        if !extent.start.get().is_multiple_of(range.bytes()) || extents.next().is_some() {
            return Err(MemoryError::Invalid(
                "atomic access is misaligned or crosses physical extents",
            ));
        }
        Ok(extent)
    }
    pub(crate) fn validate_ranges(&self, ranges: &[AllocationRange]) -> Result<usize, MemoryError> {
        ranges.iter().try_fold(0usize, |size, range| {
            if range.allocation().machine().identity() != self.identity() {
                return Err(MemoryError::Invalid(
                    "physical access names another machine",
                ));
            }
            size.checked_add(range.bytes() as usize)
                .ok_or(MemoryError::Invalid("physical capture size overflows"))
        })
    }
    /// Runs one naturally aligned host atomic operation on canonical RAM. Host
    /// CAS arbitrates with native writers; no Rust guest reference is formed.
    pub fn atomic_physical(
        &self,
        range: &AllocationRange,
        operation: crate::AtomicAccess,
    ) -> Result<crate::AccessResult, MemoryError> {
        Ok(self
            .atomic_impl(range, operation, false)?
            .expect("owner-resolved atomic cannot defer to itself"))
    }
    fn atomic_impl(
        &self,
        range: &AllocationRange,
        operation: crate::AtomicAccess,
        route_device: bool,
    ) -> Result<Option<crate::AccessResult>, MemoryError> {
        if u64::from(operation.bytes) != range.bytes() {
            return Err(MemoryError::Invalid(
                "atomic operation width differs from its captured physical span",
            ));
        }
        let extent = self.atomic_extent(range)?;
        if route_device && self.must_flush_physical(extent) {
            return Ok(None);
        }
        operation.before();
        let apply = || {
            let address = self
                .arena()
                .host_address(extent.start, extent.bytes)
                .unwrap();
            let mut previous =
                unsafe { crate::raw::read_value(address.as_ptr(), extent.bytes as usize) };
            loop {
                let (replacement, written) = operation.update(previous);
                if !written {
                    return crate::AccessResult::Atomic {
                        previous,
                        written: false,
                    };
                }

                let observed = unsafe {
                    crate::raw::compare_exchange(
                        address.as_ptr(),
                        extent.bytes as usize,
                        previous,
                        replacement,
                    )
                };
                if observed == previous {
                    return crate::AccessResult::Atomic {
                        previous,
                        written: true,
                    };
                }
                previous = observed;
            }
        };
        let result = Some(apply());
        operation.after();
        Ok(result)
    }

    // Callers borrow a live allocation or validated pins through the transition.
    // Only an operation escaping to the device owner needs independent retention.

    /// Apply the operation directly after the graphics owner has resolved its
    /// exact authority and retained its interval dependency. No endpoint reentry.
    pub fn apply_resolved_access(
        &self,
        ranges: &[AllocationRange],
        operation: &mut crate::AccessOperation,
    ) -> Result<crate::AccessResult, MemoryError> {
        match operation {
            crate::AccessOperation::Flush(_) => Err(MemoryError::Invalid(
                "rasterizer flush requires the graphics owner",
            )),

            crate::AccessOperation::Read(bytes) => {
                let size = self.validate_ranges(ranges)?;
                if bytes.len() != size {
                    bytes.resize(size, 0);
                }
                self.capture_stream(ranges, bytes)?;
                Ok(crate::AccessResult::Read(std::mem::take(bytes)))
            }
            crate::AccessOperation::Write(bytes) => {
                self.write_resolved(ranges, bytes)?;
                Ok(crate::AccessResult::Written(std::mem::take(bytes)))
            }
            crate::AccessOperation::Atomic(operation) => {
                let [range] = ranges else {
                    return Err(MemoryError::Invalid(
                        "atomic access must name one physical span",
                    ));
                };
                self.atomic_physical(range, *operation)
            }
        }
    }
    /// Complete an old captured operation after native admission ends. Physical
    /// protection may have changed since capture; routing and RAM execution are
    /// decided under access exclusion, before any write is performed.
    pub fn access_captured(
        &self,
        ranges: &[AllocationRange],
        operation: crate::AccessOperation,
    ) -> Result<crate::Access<crate::AccessResult>, MemoryError> {
        if matches!(
            operation,
            crate::AccessOperation::Write(_) | crate::AccessOperation::Atomic(_)
        ) {
            self.notify_cpu_write(ranges.iter().flat_map(AllocationRange::physical_extents));
        }
        self.finish_captured_access(ranges, operation)
    }
    fn finish_captured_access(
        &self,
        ranges: &[AllocationRange],
        mut operation: crate::AccessOperation,
    ) -> Result<crate::Access<crate::AccessResult>, MemoryError> {
        if let Some(result) = self.try_access_captured(ranges, &mut operation)? {
            return Ok(crate::Access::Ready(result));
        }
        self.request_device_access(ranges.to_vec(), operation)
            .map(crate::Access::Pending)
    }
    /// Execute admitted RAM operations directly, without submitting to a device
    /// endpoint. The graphics owner uses None to enter its exact authority in
    /// place; external callers submit that same still-unexecuted operation.
    pub fn try_access_captured(
        &self,
        ranges: &[AllocationRange],
        operation: &mut crate::AccessOperation,
    ) -> Result<Option<crate::AccessResult>, MemoryError> {
        let size = self.validate_ranges(ranges)?;
        if ranges.is_empty() {
            return Err(MemoryError::Invalid(
                "captured access requires physical spans",
            ));
        }
        match operation {
            crate::AccessOperation::Flush(_) => return Ok(None),

            crate::AccessOperation::Write(bytes) => {
                if self.write_physical_impl(ranges, bytes, true)? {
                    return Ok(Some(crate::AccessResult::Written(std::mem::take(bytes))));
                }
            }
            crate::AccessOperation::Atomic(atomic) => {
                let [range] = ranges else {
                    return Err(MemoryError::Invalid(
                        "atomic access must name one physical span",
                    ));
                };
                if let Some(result) = self.atomic_impl(range, *atomic, true)? {
                    return Ok(Some(result));
                }
            }
            crate::AccessOperation::Read(destination) => {
                if destination.len() != size {
                    destination.resize(size, 0);
                }
                if self.read_physical_into(ranges, destination)? {
                    return Ok(Some(crate::AccessResult::Read(std::mem::take(destination))));
                }
            }
        }
        Ok(None)
    }
    /// CPU ReadBlock for instruction inputs. Cached device pages must pass
    /// through the owning buffer/texture caches before filling the CPU page.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp
    pub fn read_cached_captured(
        &self,
        ranges: &[AllocationRange],
    ) -> Result<crate::Access<crate::AccessResult>, MemoryError> {
        self.validate_ranges(ranges)?;
        self.request_device_access(ranges.to_vec(), crate::AccessOperation::Read(Vec::new()))
            .map(crate::Access::Pending)
    }
    /// Copy a previously accepted span into caller-owned host storage. Pending
    /// completion owns its byte result, never a pointer into this destination.
    pub fn read_captured_into(
        &self,
        ranges: &crate::CapturedMemory,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<crate::Access<()>, MemoryError> {
        self.validate_capture(ranges)?;
        let bytes = destination.len() as u64;
        let extents = ranges.physical_extents(offset, bytes)?;
        if !self.read_ram_extents(extents, destination)? {
            return self
                .request_device_access(
                    ranges.capture(offset, bytes)?,
                    crate::AccessOperation::Read(Vec::new()),
                )
                .map(crate::Access::Pending);
        }
        Ok(crate::Access::Ready(()))
    }
    /// Move caller-owned storage across a device dependency and return that
    /// same allocation after the exact physical read completes.
    pub fn read_captured_owned(
        &self,
        ranges: &crate::CapturedMemory,
        offset: u64,
        mut destination: Vec<u8>,
    ) -> Result<crate::Access<Vec<u8>>, MemoryError> {
        self.validate_capture(ranges)?;
        let bytes = destination.len() as u64;
        if self.read_ram_extents(ranges.physical_extents(offset, bytes)?, &mut destination)? {
            return Ok(crate::Access::Ready(destination));
        }
        self.request_device_access(
            ranges.capture(offset, bytes)?,
            crate::AccessOperation::Read(destination),
        )
        .map(crate::Access::Pending)
    }
    fn validate_capture(&self, ranges: &crate::CapturedMemory) -> Result<(), MemoryError> {
        if ranges.machine.is_some_and(|id| id != self.identity()) {
            return Err(MemoryError::Invalid(
                "physical access names another machine",
            ));
        }
        Ok(())
    }
    fn read_physical_into(
        &self,
        ranges: &[AllocationRange],
        destination: &mut [u8],
    ) -> Result<bool, MemoryError> {
        if self.validate_ranges(ranges)? != destination.len() {
            return Err(MemoryError::Invalid(
                "physical read destination size mismatch",
            ));
        }
        self.read_ram_extents(
            ranges.iter().flat_map(AllocationRange::physical_extents),
            destination,
        )
    }
    // Every caller retains the validated physical spans through an allocation,
    // mapping or accepted pin for the entire copy.
    fn read_ram_extents(
        &self,
        extents: impl Iterator<Item = Extent>,
        destination: &mut [u8],
    ) -> Result<bool, MemoryError> {
        let mut offset = 0;
        for extent in extents {
            let bytes = extent.bytes as usize;
            // SAFETY: the caller retains each selected physical span.
            if !unsafe {
                self.try_read_mapped_ram(extent.start, &mut destination[offset..offset + bytes])?
            } {
                return Ok(false);
            }
            offset += bytes;
        }
        Ok(true)
    }
    /// Copy mapped, RAM-current bytes directly to their consumer. Device-owned
    /// pages return false without reading or changing the destination. This
    /// never enqueues a request to the graphics owner's own endpoint.
    ///
    /// # Safety
    /// The caller must retain all physical bytes for the duration of the call,
    /// using a borrowed Allocation, accepted AllocationRange values or a mapping lock whose
    /// mapping owner retains those ranges. Destination must be exclusively
    /// owned host memory, disjoint from guest RAM. GPU mapping metadata alone
    /// is weak and is not sufficient lifetime protection.
    pub unsafe fn try_read_mapped_ram(
        &self,
        start: ArenaOffset,
        destination: &mut [u8],
    ) -> Result<bool, MemoryError> {
        let extent = Extent {
            start,
            bytes: destination.len() as u64,
        };
        let source = self
            .arena()
            .host_address(start, extent.bytes)
            .ok_or(MemoryError::Invalid("mapped read exceeds physical RAM"))?;
        if self.must_flush_physical(extent) {
            return Ok(false);
        }
        // Coherence belongs to the real cache owners; retained inputs and the
        // fixed RAM arena provide lifetime, without a generic observation lease.
        unsafe {
            self.copy_ram_to_host(source, destination);
        }
        Ok(true)
    }
    /// Read selected retained RAM into its consumer. Cache tracking is owned by
    /// the SMMU buffer/image counters, independently of this data copy.
    #[inline]
    pub fn read_resources_scattered(
        &self,
        inputs: &[AllocationRange],
        copies: &[PhysicalCapture],
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        let reader = ResourceReader::new(self, inputs)?;
        for copy in copies {
            reader.validate(copy.source)?;
            if copy
                .destination
                .checked_add(copy.source.bytes as usize)
                .is_none_or(|end| end > destination.len())
            {
                return Err(MemoryError::Invalid("physical capture exceeds destination"));
            }
        }
        self.read_resources_with(reader, |reader| {
            for copy in copies {
                let end = copy.destination + copy.source.bytes as usize;
                reader
                    .validated_region(copy.source)
                    .copy(0, &mut destination[copy.destination..end])?;
            }
            Ok(())
        })
    }
    /// Capture layouts directly while their physical inputs remain admitted.
    /// The graphics owner resolves native producers before entering. The reader
    /// permits tiled imports without building a list of every 16-byte copy.
    pub fn read_resources<T>(
        &self,
        inputs: &[AllocationRange],
        capture: impl FnOnce(&ResourceReader<'_>) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        let reader = ResourceReader::new(self, inputs)?;
        self.read_resources_with(reader, capture)
    }
    fn read_resources_with<T>(
        &self,
        reader: ResourceReader<'_>,
        capture: impl FnOnce(&ResourceReader<'_>) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        let result = capture(&reader);
        #[cfg(any(test, feature = "performance-counters"))]
        self.tracking()
            .captured_bytes
            .fetch_add(reader.captured.get(), Ordering::Relaxed);
        result
    }
    /// Publish into retained physical ranges after producer/version selection.
    /// Cache ownership and guest fences order publication, as in Eden.
    pub fn write_resources<T>(
        &self,
        inputs: &[AllocationRange],
        publish: impl FnOnce(&ResourceWriter<'_>) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        let retained = ResourceReader::new(self, inputs)?.retained;
        publish(&ResourceWriter {
            machine: self,
            retained,
        })
    }

    /// Streaming captures keep dirty RAM dirty and preserve native protection. Callers supply the owned upload destination.
    pub fn capture_stream(
        &self,
        ranges: &[AllocationRange],
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        if self.validate_ranges(ranges)? != destination.len() {
            return Err(MemoryError::Invalid(
                "physical capture destination size mismatch",
            ));
        }
        // SAFETY: the caller's ranges retain all of these physical extents.
        unsafe {
            self.capture_stream_extents(
                ranges.iter().flat_map(AllocationRange::physical_extents),
                destination,
            )
        }
    }
    /// # Safety
    /// Every source extent is retained until this synchronous copy finishes,
    /// belongs to this machine, and their total length equals the destination.
    pub(crate) unsafe fn capture_stream_extents(
        &self,
        extents: impl Iterator<Item = Extent>,
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        let mut offset = 0;
        for extent in extents {
            let bytes = extent.bytes as usize;
            // SAFETY: the caller retains the extent and admission is held.
            unsafe {
                self.capture_stream_admitted(extent, &mut destination[offset..offset + bytes]);
            }
            offset += bytes;
        }
        Ok(())
    }
    /// # Safety
    /// The extent must stay live through this synchronous read, physical
    /// admission must cover it, and the destination must match its size.
    pub(crate) unsafe fn capture_stream_admitted(&self, extent: Extent, destination: &mut [u8]) {
        debug_assert_eq!(extent.bytes, destination.len() as u64);
        let source = self
            .arena()
            .host_address(extent.start, extent.bytes)
            .expect("retained physical extent");
        unsafe {
            self.copy_ram_to_host(source, destination);
        }
    }
    /// Copy raw RAM after the caller resolves coherence. The physical arena
    /// remains mapped for the machine lifetime, independently of device maps.
    ///
    /// # Safety
    /// Source is readable RAM in this machine; destination is disjoint host
    /// storage. No retained Rust references may alias the guest bytes.
    pub(crate) unsafe fn copy_ram_to_host(
        &self,
        source: crate::HostAddress,
        destination: &mut [u8],
    ) {
        unsafe {
            crate::raw::capture(source.as_ptr(), destination);
        }
        #[cfg(any(test, feature = "performance-counters"))]
        self.tracking()
            .captured_bytes
            .fetch_add(destination.len() as u64, Ordering::Relaxed);
    }
    /// Select completed download intervals under the actual buffer-cache owner.
    /// Physical pins retain the destination; only selected bytes are published.
    pub fn write_download(
        &self,
        range: &AllocationRange,
        source: &[u8],
        select: impl FnOnce() -> Result<Vec<std::ops::Range<usize>>, MemoryError>,
    ) -> Result<(), MemoryError> {
        if range.bytes() != source.len() as u64 {
            return Err(MemoryError::Invalid("download payload size mismatch"));
        }
        self.validate_ranges(std::slice::from_ref(range))?;
        let extents: Vec<_> = range.physical_extents().collect();
        let apply = || {
            let selected = select()?;
            if selected
                .iter()
                .any(|part| part.start > part.end || part.end > source.len())
            {
                return Err(MemoryError::Invalid(
                    "download selection exceeds its destination",
                ));
            }
            let mut base = 0usize;
            for extent in &extents {
                let end = base + extent.bytes as usize;
                for part in &selected {
                    let begin = base.max(part.start);
                    let stop = end.min(part.end);
                    if begin < stop {
                        let target = Extent {
                            start: ArenaOffset::new(extent.start.get() + (begin - base) as u64),
                            bytes: (stop - begin) as u64,
                        };
                        // The input retains this exact physical destination.
                        unsafe {
                            self.write_admitted(target, &source[begin..stop]);
                        }
                    }
                }
                base = end;
            }
            Ok(())
        };
        apply()
    }
    /// The graphics owner has already resolved exact authority/dependencies.
    /// Apply the owned write before releasing its interval dependency. This
    /// method never sends to the owner's own inbox.
    pub fn write_resolved(
        &self,
        ranges: &[AllocationRange],
        source: &[u8],
    ) -> Result<(), MemoryError> {
        let written = self.write_physical_impl(ranges, source, false)?;
        debug_assert!(written);
        Ok(())
    }
    /// The caller has left native and host access admission. Recheck physical
    /// device summaries under short access exclusion: a first GPU writer may
    /// have appeared after the old virtual operation was captured.
    pub fn write_captured(
        &self,
        ranges: &crate::CapturedMemory,
        offset: u64,
        source: &[u8],
    ) -> Result<crate::Access<()>, MemoryError> {
        if self.write_captured_ram(ranges, offset, source)? {
            return Ok(crate::Access::Ready(()));
        }
        self.request_device_access(
            ranges.capture(offset, source.len() as u64)?,
            crate::AccessOperation::Write(source.into()),
        )
        .map(crate::Access::Pending)
    }
    /// The pending request owns the service's buffer rather than a snapshot of
    /// a still-retained source. Completion returns it for the next bulk chunk.
    pub fn write_captured_owned(
        &self,
        ranges: &crate::CapturedMemory,
        offset: u64,
        source: Vec<u8>,
    ) -> Result<crate::Access<Vec<u8>>, MemoryError> {
        if self.write_captured_ram(ranges, offset, &source)? {
            return Ok(crate::Access::Ready(source));
        }
        self.request_device_access(
            ranges.capture(offset, source.len() as u64)?,
            crate::AccessOperation::Write(source),
        )
        .map(crate::Access::Pending)
    }
    fn write_captured_ram(
        &self,
        ranges: &crate::CapturedMemory,
        offset: u64,
        source: &[u8],
    ) -> Result<bool, MemoryError> {
        self.validate_capture(ranges)?;
        let extents: smallvec::SmallVec<[_; 4]> = ranges
            .physical_extents(offset, source.len() as u64)?
            .collect();
        self.notify_cpu_write(extents.iter().copied());
        if source.is_empty() {
            return Ok(true);
        }
        Ok(self.try_write_extents(&extents, source, true))
    }
    fn write_physical_impl(
        &self,
        ranges: &[AllocationRange],
        source: &[u8],
        route_device: bool,
    ) -> Result<bool, MemoryError> {
        if self.validate_ranges(ranges)? != source.len() {
            return Err(MemoryError::Invalid("physical write payload size mismatch"));
        }
        let extents: smallvec::SmallVec<[_; 4]> = ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .collect();
        Ok(self.try_write_extents(&extents, source, route_device))
    }

    /// Inputs and payload have already been validated and retained. None needs
    /// an observation transition; Some(false) needs device-owner resolution.
    fn try_write_extents(&self, extents: &[Extent], source: &[u8], route_device: bool) -> bool {
        if route_device
            && extents
                .iter()
                .any(|&extent| self.must_flush_physical(extent))
        {
            return false;
        }

        let mut offset = 0;
        for &extent in extents {
            let end = offset + extent.bytes as usize;
            // SAFETY: the caller retains every destination through the stores.
            unsafe {
                self.write_admitted(extent, &source[offset..end]);
            }
            offset = end;
        }
        true
    }
}

impl crate::Allocation {
    /// Access RAM using this owner's lifetime. Only a deferred device operation
    /// retains an independent exact range.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_shared_memory.h
    pub fn read(&self, offset: u64, bytes: usize) -> Result<crate::Access<Box<[u8]>>, MemoryError> {
        if bytes == 0 {
            return Ok(crate::Access::Ready(Box::new([])));
        }
        let extents = self.physical_extents(offset, bytes as u64)?;
        let mut output = vec![0; bytes];
        if !self.machine().read_ram_extents(extents, &mut output)? {
            let range =
                AllocationRange::new(self.pin_range(offset, bytes as u64)?, offset, bytes as u64)?;
            return self
                .machine()
                .request_device_access(vec![range], crate::AccessOperation::Read(Vec::new()))
                .map(crate::Access::Pending);
        }
        Ok(crate::Access::Ready(output.into_boxed_slice()))
    }
    /// Write through the existing owner, preserving observation transitions and
    /// deferring device-dependent writes intact before modifying any RAM prefix.
    pub fn write(&self, offset: u64, source: &[u8]) -> Result<crate::Access<()>, MemoryError> {
        if source.is_empty() {
            return Ok(crate::Access::Ready(()));
        }
        let extents: smallvec::SmallVec<[_; 4]> = self
            .physical_extents(offset, source.len() as u64)?
            .collect();
        let machine = self.machine();
        machine.notify_cpu_write(extents.iter().copied());
        let written = machine.try_write_extents(&extents, source, true);
        if written {
            return Ok(crate::Access::Ready(()));
        }
        let range = AllocationRange::new(
            self.pin_range(offset, source.len() as u64)?,
            offset,
            source.len() as u64,
        )?;
        machine
            .request_device_access(vec![range], crate::AccessOperation::Write(source.into()))
            .map(crate::Access::Pending)
    }
}
