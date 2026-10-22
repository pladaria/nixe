//! Physical dirty summaries and independently invalidated clean observations.
use crate::{AddressSpace, AllocationRange, ArenaOffset, Extent, Machine, MemoryError, PAGE_SIZE};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

struct Bitmap(Box<[AtomicU64]>);
impl Bitmap {
    fn new(pages: u64, value: u64) -> Self {
        Self(
            (0..pages.div_ceil(64))
                .map(|_| AtomicU64::new(value))
                .collect(),
        )
    }
    fn get(&self, page: u64) -> bool {
        self.0[(page / 64) as usize].load(Ordering::Acquire) & (1 << (page % 64)) != 0
    }
    fn any(&self, extent: Extent) -> bool {
        bitmap_words(extent).any(|(word, mask)| self.0[word].load(Ordering::Acquire) & mask != 0)
    }
    fn update_range(&self, extent: Extent, value: bool, mut cleared: impl FnMut(u64)) {
        for (word, mask) in bitmap_words(extent) {
            if value {
                self.0[word].fetch_or(mask, Ordering::Release);
            } else {
                let mut previous = self.0[word].fetch_and(!mask, Ordering::AcqRel) & mask;
                while previous != 0 {
                    let bit = previous.trailing_zeros();
                    cleared(word as u64 * 64 + u64::from(bit));
                    previous &= previous - 1;
                }
            }
        }
    }
}
pub(crate) struct Tracking {
    observed: Bitmap,
    device_observed: Bitmap,
    dirty: Bitmap,
    reserved: Bitmap,
    device: Bitmap,
    code: Bitmap,
    monitors: crate::reservation::Monitors,
    observers: Mutex<Observers>,
    cpu_dirty_pending: AtomicBool,
    protection_calls: AtomicU64,
    #[cfg(any(test, feature = "performance-counters"))]
    captured_bytes: AtomicU64,
}
#[derive(Default)]
struct Observers {
    pages: HashMap<u64, PageObservers>,
    dirty_words: Vec<usize>,
}
// Cached-page counts do not revisit existing consumers on every increment.
// Weak flags also need reclamation; amortize it over insertions here.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
#[derive(Default)]
struct PageObservers {
    entries: Vec<Observer>,
    prune_at: usize,
    #[cfg(test)]
    registration_inspections: usize,
}
impl PageObservers {
    fn retain(&mut self, keep: impl FnMut(&Observer) -> bool) {
        self.entries.retain(keep);
        // Charge the next sweep to at least as many insertions as survivors.
        // Use an insertion budget, not Vec capacity: reclaiming just one dead
        // entry from a full Vec must not cause another full scan on the next push.
        self.prune_at = self.entries.len().saturating_mul(2).max(16);
    }
    fn push(&mut self, observer: Observer) {
        if self.entries.len() >= self.prune_at {
            #[cfg(test)]
            {
                self.registration_inspections += self.entries.len();
            }
            self.retain(|observer| {
                observer
                    .invalid
                    .upgrade()
                    .is_some_and(|invalid| !invalid.load(Ordering::Acquire))
            });
        }
        self.entries.push(observer);
    }
}
struct Observer {
    invalid: Weak<AtomicBool>,
    cpu_writes: bool,
    device_writes: bool,
}
/// A cache checks this flag once per use, without walking its physical pages.
/// The flag never becomes clean again; a new capture creates a new observation.
#[derive(Clone, Debug)]
pub struct Observation(Arc<AtomicBool>);
impl Default for Observation {
    fn default() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
}
impl Observation {
    pub fn is_invalid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn invalidate(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Host metadata only. Subscribers may set this flag, never clear it.
    pub fn weak_invalidation_flag(&self) -> Weak<AtomicBool> {
        Arc::downgrade(&self.0)
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
    pub bitmap_bytes: usize,
    pub observer_pages: usize,
    pub observer_registrations: usize,
    pub dirty_queue_bytes: usize,
    pub protection_calls: u64,
    pub captured_bytes: u64,
    pub reservation_metadata_bytes: usize,
}
struct AliasRange {
    space: usize,
    start: u64,
    bytes: u64,
}
struct ProtectionBatch<'a> {
    aliases: Vec<AliasRange>,
    spaces: &'a [Arc<AddressSpace>],
}
impl ProtectionBatch<'_> {
    fn invalidate(&self, machine: &Machine, extents: &[Extent]) {
        self.invalidate_from(machine, extents, false);
    }
    fn invalidate_from(&self, machine: &Machine, extents: &[Extent], device: bool) {
        if extents
            .iter()
            .any(|&extent| machine.tracking().code.any(extent))
        {
            for alias in &self.aliases {
                // SAFETY: all affected spaces have drained native execution and
                // host accesses. Queue invalidations before clearing code watches.
                unsafe {
                    self.spaces[alias.space].invalidate_code_writes(alias.start, alias.bytes);
                }
            }
        }
        if device {
            machine.tracking().invalidate_device(extents);
        } else {
            machine.tracking().invalidate(extents);
        }
    }
    fn refresh(&self) -> Result<(), MemoryError> {
        for alias in &self.aliases {
            // SAFETY: alias and space writers exclude mapping mutation. Host and
            // checked physical accesses have drained; tracking fields are atomic.
            unsafe {
                self.spaces[alias.space].refresh_tracking(alias.start, alias.bytes)?;
            }
        }
        Ok(())
    }
}
fn pages(extent: Extent) -> std::ops::Range<u64> {
    extent.start.get() / PAGE_SIZE..(extent.start.get() + extent.bytes).div_ceil(PAGE_SIZE)
}
fn bitmap_words(extent: Extent) -> impl Iterator<Item = (usize, u64)> {
    let pages = pages(extent);
    (pages.start / 64..pages.end.div_ceil(64)).map(move |word| {
        let low = pages.start.saturating_sub(word * 64);
        let high = (pages.end - word * 64).min(64);
        (word as usize, (u64::MAX << low) & (u64::MAX >> (64 - high)))
    })
}
impl Tracking {
    pub fn new(bytes: u64) -> Self {
        Self {
            observed: Bitmap::new(bytes / PAGE_SIZE, 0),
            device_observed: Bitmap::new(bytes / PAGE_SIZE, 0),
            dirty: Bitmap::new(bytes / PAGE_SIZE, 0),
            reserved: Bitmap::new(bytes / PAGE_SIZE, 0),
            device: Bitmap::new(bytes / PAGE_SIZE, 0),
            code: Bitmap::new(bytes / PAGE_SIZE, 0),
            monitors: crate::reservation::Monitors::default(),
            observers: Mutex::new(Observers::default()),
            cpu_dirty_pending: AtomicBool::new(false),
            protection_calls: AtomicU64::new(0),
            #[cfg(any(test, feature = "performance-counters"))]
            captured_bytes: AtomicU64::new(0),
        }
    }
    pub fn observed(&self, page: u64) -> bool {
        self.observed.get(page)
    }
    pub fn code(&self, page: u64) -> bool {
        self.code.get(page)
    }
    pub fn native_flags(&self, page: u64) -> u64 {
        if !self.reserved.get(page) {
            return 1;
        }
        // The caller retains an admitted mapping or a captured allocation.
        std::ptr::from_ref(
            unsafe { self.monitors.get(page) }.expect("published physical reservation monitor"),
        ) as u64
    }
    pub fn retire(&self, extents: &[Extent]) {
        self.invalidate(extents);
        self.invalidate_device(extents);
        for &extent in extents {
            self.device.update_range(extent, false, |_| {});
            self.reserved
                .update_range(extent, false, |page| self.monitors.remove(page));
        }
    }
    pub fn kind(&self, page: u64) -> u8 {
        crate::address_space::MappedPage::RAM
            | if self.observed(page) {
                crate::address_space::MappedPage::OBSERVED
            } else {
                0
            }
            | if self.device.get(page) {
                crate::address_space::MappedPage::DEVICE
            } else {
                0
            }
    }
    pub fn record_protection(&self) {
        self.protection_calls.fetch_add(1, Ordering::Relaxed);
    }
    // Caller excludes physical writers, or owns the last allocation pin before
    // recycling. Invalidation precedes both the store and direct write access.
    pub fn invalidate(&self, extents: &[Extent]) {
        let mut observers = self.observers.lock().unwrap();
        for &extent in extents {
            self.code.update_range(extent, false, |_| {});
            self.observed.update_range(extent, false, |page| {
                let mut device_only = false;
                if let Some(registered) = observers.pages.get_mut(&page) {
                    registered.retain(|observer| {
                        let Some(invalid) = observer.invalid.upgrade() else {
                            return false;
                        };
                        if observer.cpu_writes {
                            invalid.store(true, Ordering::Release);
                            false
                        } else {
                            let live = !invalid.load(Ordering::Acquire);
                            device_only |= live && observer.device_writes;
                            live
                        }
                    });
                    if registered.entries.is_empty() {
                        observers.pages.remove(&page);
                    }
                }
                self.device_observed.update_range(
                    Extent {
                        start: ArenaOffset::new(page * PAGE_SIZE),
                        bytes: PAGE_SIZE,
                    },
                    device_only,
                    |_| {},
                );
            });
            for (word, mask) in bitmap_words(extent) {
                if self.dirty.0[word].fetch_or(mask, Ordering::Release) == 0 {
                    observers.dirty_words.push(word);
                }
            }
        }
        if !extents.is_empty() {
            self.cpu_dirty_pending.store(true, Ordering::Release);
        }
    }
    fn invalidate_device(&self, extents: &[Extent]) {
        let mut observers = self.observers.lock().unwrap();
        for &extent in extents {
            self.code.update_range(extent, false, |_| {});
            self.device_observed.update_range(extent, false, |page| {
                let empty = if let Some(registered) = observers.pages.get_mut(&page) {
                    registered.retain(|observer| {
                        let Some(invalid) = observer.invalid.upgrade() else {
                            return false;
                        };
                        if observer.device_writes {
                            invalid.store(true, Ordering::Release);
                            false
                        } else {
                            !invalid.load(Ordering::Acquire)
                        }
                    });
                    registered.entries.is_empty()
                } else {
                    true
                };
                if empty {
                    observers.pages.remove(&page);
                    self.observed.update_range(
                        Extent {
                            start: ArenaOffset::new(page * PAGE_SIZE),
                            bytes: PAGE_SIZE,
                        },
                        false,
                        |_| {},
                    );
                }
            });
        }
    }
    fn register(
        &self,
        extents: impl Iterator<Item = Extent>,
        observation: &Observation,
        cpu_writes: bool,
        device_writes: bool,
    ) {
        let mut observers = self.observers.lock().unwrap();
        for extent in extents {
            for page in pages(extent) {
                if device_writes {
                    self.device_observed.update_range(
                        Extent {
                            start: ArenaOffset::new(page * PAGE_SIZE),
                            bytes: PAGE_SIZE,
                        },
                        true,
                        |_| {},
                    );
                }
                let registered = observers.pages.entry(page).or_default();
                registered.push(Observer {
                    invalid: Arc::downgrade(&observation.0),
                    cpu_writes,
                    device_writes,
                });
            }
        }
    }
    fn statistics(&self) -> TrackingStatistics {
        let observers = self.observers.lock().unwrap();
        TrackingStatistics {
            bitmap_bytes: self.observed.0.len() * 48,
            observer_pages: observers.pages.len(),
            observer_registrations: observers
                .pages
                .values()
                .map(|page| page.entries.len())
                .sum(),
            dirty_queue_bytes: observers.dirty_words.capacity() * std::mem::size_of::<usize>(),
            protection_calls: self.protection_calls.load(Ordering::Relaxed),
            #[cfg(any(test, feature = "performance-counters"))]
            captured_bytes: self.captured_bytes.load(Ordering::Relaxed),
            #[cfg(not(any(test, feature = "performance-counters")))]
            captured_bytes: 0,
            reservation_metadata_bytes: self.monitors.bytes(),
        }
    }
}
impl Machine {
    /// The graphics owner consumes CPU modification pages once, then updates all
    /// registered views before resource capture. These are cache invalidations;
    /// exact CPU/device byte ownership remains with the operation authority.
    pub fn take_cpu_modified_pages(&self) -> Vec<Extent> {
        let tracking = self.tracking();
        if !tracking.cpu_dirty_pending.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut observers = tracking.observers.lock().unwrap();
        let mut result: Vec<Extent> = Vec::new();
        for &word in &observers.dirty_words {
            let mut bits = tracking.dirty.0[word].swap(0, Ordering::AcqRel);
            while bits != 0 {
                let low = bits.trailing_zeros();
                let high = low + (bits >> low).trailing_ones();
                let start = (word as u64 * 64 + u64::from(low)) * PAGE_SIZE;
                let bytes = u64::from(high - low) * PAGE_SIZE;
                if let Some(previous) = result
                    .last_mut()
                    .filter(|previous| previous.start.get() + previous.bytes == start)
                {
                    previous.bytes += bytes;
                } else {
                    result.push(Extent {
                        start: ArenaOffset::new(start),
                        bytes,
                    });
                }
                bits = if high == 64 {
                    0
                } else {
                    bits & (u64::MAX << high)
                };
            }
        }
        observers.dirty_words.clear();
        tracking.cpu_dirty_pending.store(false, Ordering::Release);
        result
    }
}

impl Machine {
    /// # Safety
    /// Caller retains/admitted this exact physical span and has resolved its
    /// observed/device dependencies. Ordinary stores need no reservation lookup;
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
    /// Retained producers already own device-affected pages. Invalidate only
    /// newly observed input pages before appending their next native write;
    /// membership and physical reservation epochs are managed by the producer.
    pub fn invalidate_device_observations<T>(
        &self,
        ranges: &[AllocationRange],
        publish: impl FnOnce() -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        self.validate_ranges(ranges)?;
        if ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .all(|extent| !self.tracking().device_observed.any(extent))
        {
            return publish();
        }
        self.tracking_transition(ranges, |batch, extents| {
            if extents
                .iter()
                .any(|&extent| pages(extent).any(|page| !self.tracking().device.get(page)))
            {
                return Err(MemoryError::Invalid(
                    "observed producer must establish device coverage before reusing it",
                ));
            }
            batch.invalidate_from(self, extents, true);
            batch.refresh()?;
            publish()
        })
    }
    /// The owner calls this when exact device coverage first enters physical
    /// pages. Append the producing command and publish its coverage inside the
    /// callback, before CPU admission reopens. A retained coverage record may
    /// skip page publication on repeated writes, but must still invalidate any
    /// new CPU/code observations ordered through the owner since its last write.
    pub fn publish_device_write<T>(
        &self,
        ranges: &[AllocationRange],
        publish: impl FnOnce() -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        self.validate_ranges(ranges)?;
        if ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .all(|extent| {
                pages(extent).all(|page| {
                    self.tracking().device.get(page) && !self.tracking().device_observed.get(page)
                })
            })
        {
            self.mark_device_reservations(ranges);
            return publish();
        }
        self.tracking_transition(ranges, |batch, extents| {
            batch.invalidate_from(self, extents, true);
            for &extent in extents {
                self.tracking().device.update_range(extent, true, |_| {});
            }
            batch.refresh()?;
            self.mark_device_reservations(ranges);
            publish()
        })
    }
    fn mark_device_reservations(&self, ranges: &[AllocationRange]) {
        if let [range] = ranges {
            range.allocation().mark_device_modified();
        } else {
            let mut seen = std::collections::HashSet::new();
            for range in ranges {
                if seen.insert(range.allocation().identity()) {
                    range.allocation().mark_device_modified();
                }
            }
        }
    }
    /// The exact authority index calls this only after its final coverage has
    /// left every supplied page. Other producers in a shared page must keep
    /// that page protected; memory does not maintain another producer index.
    pub fn retire_device_pages(&self, ranges: &[AllocationRange]) -> Result<(), MemoryError> {
        self.validate_ranges(ranges)?;
        if ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .all(|extent| pages(extent).all(|page| !self.tracking().device.get(page)))
        {
            return Ok(());
        }
        self.tracking_transition(ranges, |batch, extents| {
            for &extent in extents {
                self.tracking().device.update_range(extent, false, |_| {});
            }
            batch.refresh()
        })
    }
    pub fn tracking_statistics(&self) -> TrackingStatistics {
        self.tracking().statistics()
    }
    /// Activates the physical native monitor before allowing any exclusive
    /// access. Existing monitors require neither quiescence nor reprotection.
    pub fn prepare_reservations(&self, ranges: &[AllocationRange]) -> Result<(), MemoryError> {
        self.validate_ranges(ranges)?;
        if ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .all(|extent| pages(extent).all(|page| self.tracking().reserved.get(page)))
        {
            return Ok(());
        }
        self.tracking_transition_with_native(ranges, true, |batch, _| {
            for range in ranges {
                let epoch = range.allocation().device_write_epoch();
                epoch.activate();
                for extent in range.physical_extents() {
                    for page in pages(extent) {
                        let base = self
                            .arena()
                            .host_address(ArenaOffset::new(page * PAGE_SIZE), PAGE_SIZE)
                            .expect("retained reservation page");
                        self.tracking()
                            .monitors
                            .ensure(page, base.as_ptr() as usize, &epoch);
                    }
                    self.tracking().reserved.update_range(extent, true, |_| {});
                }
            }
            batch.refresh()
        })
    }
    fn exclusive_extent(&self, range: &AllocationRange) -> Result<Extent, MemoryError> {
        self.validate_ranges(std::slice::from_ref(range))?;
        if !matches!(range.bytes(), 1 | 2 | 4 | 8 | 16) {
            return Err(MemoryError::Invalid("unsupported exclusive access width"));
        }
        let mut extents = range.physical_extents();
        let extent = extents.next().expect("nonempty retained range");
        if !extent.start.get().is_multiple_of(range.bytes()) || extents.next().is_some() {
            return Err(MemoryError::Invalid(
                "exclusive access is misaligned or crosses physical extents",
            ));
        }
        Ok(extent)
    }
    /// Convert a stopped native instruction's copied proof against its captured
    /// physical destination. This performs no guest access, lock or wait and may
    /// run in a C++ callback. Apply the resulting exact operation only after Run
    /// returns and admission has suspended. The native address is never dereferenced.
    pub fn capture_native_reservation(
        &self,
        range: &AllocationRange,
        native: crate::NativeReservation,
    ) -> Result<Option<crate::Reservation>, MemoryError> {
        let extent = self.exclusive_extent(range)?;
        let canonical = self
            .arena()
            .host_address(extent.start, extent.bytes)
            .expect("retained exclusive extent");
        if native.bytes != extent.bytes || native.canonical_address != canonical.as_ptr() as u64 {
            return Ok(None);
        }
        Ok(Some(crate::Reservation {
            machine: self.identity(),
            allocation: range.allocation().identity(),
            physical: extent.start,
            bytes: extent.bytes as u8,
            monitor: native.identity,
            version: native.version,
            device_version: native.device_version,
            expected: native.expected,
        }))
    }
    /// Publish an owner-completed exclusive load into its stopped native core.
    /// This copies metadata only; the reservation never retains guest RAM.
    pub fn export_native_reservation(
        &self,
        reservation: crate::Reservation,
    ) -> Result<crate::NativeReservation, MemoryError> {
        if reservation.machine != self.identity() {
            return Err(MemoryError::Invalid(
                "native reservation names another machine",
            ));
        }
        let address = self
            .arena()
            .host_address(reservation.physical, u64::from(reservation.bytes))
            .ok_or(MemoryError::Invalid(
                "native reservation exceeds physical RAM",
            ))?;
        Ok(crate::NativeReservation {
            identity: reservation.monitor,
            version: reservation.version,
            device_version: reservation.device_version,
            canonical_address: address.as_ptr() as u64,
            bytes: u64::from(reservation.bytes),
            expected: reservation.expected,
        })
    }
    pub fn load_exclusive_resolved(
        &self,
        range: &AllocationRange,
    ) -> Result<(u128, crate::Reservation), MemoryError> {
        Ok(self
            .load_exclusive_impl(range, false)?
            .expect("owner-resolved exclusive load cannot defer"))
    }
    fn load_exclusive_impl(
        &self,
        range: &AllocationRange,
        route_device: bool,
    ) -> Result<Option<(u128, crate::Reservation)>, MemoryError> {
        let extent = self.exclusive_extent(range)?;
        if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
            return Ok(None);
        }
        self.prepare_reservations(std::slice::from_ref(range))?;
        let _access = self.physical_access().enter(extent);
        if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
            return Ok(None);
        }
        // The retained range pins this monitor's allocation lifetime.
        let monitor = unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
            .expect("prepared physical exclusive monitor");
        let lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
        let source = self
            .arena()
            .host_address(extent.start, extent.bytes)
            .unwrap();
        let value = unsafe { crate::raw::read_value(source.as_ptr(), extent.bytes as usize) };
        Ok(Some((
            value,
            crate::Reservation {
                machine: self.identity(),
                allocation: range.allocation().identity(),
                physical: extent.start,
                bytes: extent.bytes as u8,
                monitor: monitor.identity(),
                version: lock.version(),
                device_version: monitor.device_version(),
                expected: [value as u64, (value >> 64) as u64],
            },
        )))
    }
    pub fn store_exclusive_resolved(
        &self,
        range: &AllocationRange,
        reservation: crate::Reservation,
        value: u128,
    ) -> Result<bool, MemoryError> {
        Ok(self
            .store_exclusive_impl(range, reservation, value, false)?
            .expect("owner-resolved exclusive store cannot defer"))
    }
    fn store_exclusive_impl(
        &self,
        range: &AllocationRange,
        reservation: crate::Reservation,
        value: u128,
        route_device: bool,
    ) -> Result<Option<bool>, MemoryError> {
        let extent = self.exclusive_extent(range)?;
        if reservation.machine != self.identity()
            || reservation.allocation != range.allocation().identity()
            || reservation.physical != extent.start
            || u64::from(reservation.bytes) != extent.bytes
        {
            return Ok(Some(false));
        }
        let store = |batch: Option<&ProtectionBatch<'_>>| {
            let monitor = unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
                .expect("live reservation retains its prepared monitor");
            let mut lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
            if monitor.identity() != reservation.monitor
                || lock.version() != reservation.version
                || monitor.device_version() != reservation.device_version
            {
                return false;
            }
            if self.tracking().observed(extent.start.get() / PAGE_SIZE) {
                batch
                    .expect("observed exclusive store requires alias quiescence")
                    .invalidate(self, &[extent]);
            }
            let destination = self
                .arena()
                .host_address(extent.start, extent.bytes)
                .unwrap();
            // Ordinary native stores do not take the reservation lock. Commit
            // with the observed value, as in Eden's WriteExclusive callbacks:
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/memory.cpp
            let expected =
                u128::from(reservation.expected[0]) | (u128::from(reservation.expected[1]) << 64);
            let previous = unsafe {
                crate::raw::compare_exchange(
                    destination.as_ptr(),
                    extent.bytes as usize,
                    expected,
                    value,
                )
            };
            let written = previous == expected;
            if written {
                lock.written();
            }
            written
        };
        let access = self.physical_access().enter(extent);
        if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
            let monitor = unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
                .expect("live reservation retains its monitor");
            let lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
            if monitor.identity() != reservation.monitor
                || lock.version() != reservation.version
                || monitor.device_version() != reservation.device_version
            {
                return Ok(Some(false));
            }
            return Ok(None);
        }
        if !self.tracking().observed(extent.start.get() / PAGE_SIZE) {
            return Ok(Some(store(None)));
        }
        {
            let monitor = unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
                .expect("live reservation retains its prepared monitor");
            let lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
            if monitor.identity() != reservation.monitor
                || lock.version() != reservation.version
                || monitor.device_version() != reservation.device_version
            {
                return Ok(Some(false));
            }
            let address = self
                .arena()
                .host_address(extent.start, extent.bytes)
                .unwrap();
            let expected =
                u128::from(reservation.expected[0]) | (u128::from(reservation.expected[1]) << 64);
            // A plain store can change the value without changing this monitor.
            // Reject before disarming a subsequently established observation.
            if unsafe { crate::raw::read_value(address.as_ptr(), extent.bytes as usize) }
                != expected
            {
                return Ok(Some(false));
            }
        }
        drop(access);
        self.tracking_transition(std::slice::from_ref(range), |batch, _| {
            if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
                return Ok(None);
            }
            let written = store(Some(batch));
            batch.refresh()?;
            Ok(Some(written))
        })
    }
    fn validate_ranges(&self, ranges: &[AllocationRange]) -> Result<usize, MemoryError> {
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
    /// Runs one naturally aligned host atomic operation on canonical RAM. The
    /// reservation monitor orders exclusive/checked writers; host CAS also
    /// arbitrates with direct native stores. No Rust guest reference is formed.
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
        match operation.operation {
            crate::AtomicOperation::LoadExclusive => {
                operation.before();
                let result = self.load_exclusive_impl(range, route_device)?;
                operation.after();
                return Ok(
                    result.map(|(value, reservation)| crate::AccessResult::ExclusiveRead {
                        value,
                        reservation,
                    }),
                );
            }
            crate::AtomicOperation::StoreExclusive {
                reservation,
                replacement,
            } => {
                operation.before();
                let result =
                    self.store_exclusive_impl(range, reservation, replacement, route_device)?;
                operation.after();
                return Ok(result.map(crate::AccessResult::ExclusiveWritten));
            }
            _ => {}
        }
        let extent = self.exclusive_extent(range)?;
        if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
            return Ok(None);
        }
        self.prepare_reservations(std::slice::from_ref(range))?;
        operation.before();
        let apply = |batch: Option<&ProtectionBatch<'_>>| {
            let monitor = unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
                .expect("prepared physical atomic monitor");
            let mut lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
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
                if self.tracking().observed(extent.start.get() / PAGE_SIZE) {
                    batch
                        .expect("observed atomic write requires alias quiescence")
                        .invalidate(self, &[extent]);
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
                    lock.written();
                    return crate::AccessResult::Atomic {
                        previous,
                        written: true,
                    };
                }
                previous = observed;
            }
        };
        let access = self.physical_access().enter(extent);
        if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
            return Ok(None);
        }
        let result = if self.tracking().observed(extent.start.get() / PAGE_SIZE) {
            // A failed comparison writes nothing and leaves every clean
            // observer armed, without requesting an unnecessary interruption.
            if let crate::AtomicOperation::CompareExchange { .. } = operation.operation {
                let monitor =
                    unsafe { self.tracking().monitors.get(extent.start.get() / PAGE_SIZE) }
                        .expect("prepared physical atomic monitor");
                let _lock = monitor.lock((extent.start.get() % PAGE_SIZE) as usize);
                let address = self
                    .arena()
                    .host_address(extent.start, extent.bytes)
                    .unwrap();
                let previous =
                    unsafe { crate::raw::read_value(address.as_ptr(), extent.bytes as usize) };
                if !operation.update(previous).1 {
                    operation.after();
                    return Ok(Some(crate::AccessResult::Atomic {
                        previous,
                        written: false,
                    }));
                }
            }
            drop(access);
            self.tracking_transition(std::slice::from_ref(range), |batch, _| {
                if route_device && self.tracking().device.get(extent.start.get() / PAGE_SIZE) {
                    return Ok(None);
                }
                let result = apply(Some(batch));
                batch.refresh()?;
                Ok(Some(result))
            })?
        } else {
            Some(apply(None))
        };
        operation.after();
        Ok(result)
    }
    fn tracking_transition<T>(
        &self,
        ranges: &[AllocationRange],
        operation: impl FnOnce(&ProtectionBatch<'_>, &[Extent]) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        self.tracking_transition_with_native(ranges, false, operation)
    }
    fn tracking_transition_with_native<T>(
        &self,
        ranges: &[AllocationRange],
        force_native: bool,
        operation: impl FnOnce(&ProtectionBatch<'_>, &[Extent]) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        self.validate_ranges(ranges)?;
        self.tracking_transition_physical(
            ranges.iter().flat_map(|range| {
                range
                    .physical_extents()
                    .map(|extent| (range.allocation().identity(), extent))
            }),
            force_native,
            operation,
        )
    }
    // Callers borrow a live allocation or validated pins through the transition.
    // Only an operation escaping to the device owner needs independent retention.
    fn tracking_transition_physical<T>(
        &self,
        input: impl Iterator<Item = (crate::AllocationId, Extent)>,
        force_native: bool,
        operation: impl FnOnce(&ProtectionBatch<'_>, &[Extent]) -> Result<T, MemoryError>,
    ) -> Result<T, MemoryError> {
        // Spaces must outlive (and drop after) the index guard: their destructor
        // removes reverse mappings from that same index.
        let mut spaces = Vec::new();
        let index = self.aliases();
        let mut extents = smallvec::SmallVec::<[Extent; 4]>::new();
        let mut selected = Vec::new();
        for (allocation, extent) in input {
            extents.push(extent);
            let start = extent.start.get() & !(PAGE_SIZE - 1);
            let end = (extent.start.get() + extent.bytes).next_multiple_of(PAGE_SIZE);
            index.index.overlapping(start, end, |alias| {
                if alias.allocation != allocation {
                    return;
                }
                if let Some(space) = alias.space.upgrade() {
                    let begin = start.max(alias.physical);
                    let stop = end.min(alias.physical + alias.bytes);
                    selected.push((
                        space.identity(),
                        alias.virtual_address + begin - alias.physical,
                        stop - begin,
                    ));
                    spaces.push(space);
                }
            });
        }
        let interrupt_native = force_native
            || extents
                .iter()
                .any(|&extent| self.tracking().code.any(extent));
        spaces.sort_unstable_by_key(|space| space.identity());
        spaces.dedup_by_key(|space| space.identity());
        selected.sort_unstable();
        let mut aliases: Vec<AliasRange> = Vec::new();
        for (id, start, bytes) in selected {
            let space = spaces
                .binary_search_by_key(&id, |space| space.identity())
                .unwrap();
            if let Some(last) = aliases
                .last_mut()
                .filter(|last| last.space == space && start <= last.start + last.bytes)
            {
                last.bytes = (start + bytes).max(last.start + last.bytes) - last.start;
            } else {
                aliases.push(AliasRange {
                    space,
                    start,
                    bytes,
                });
            }
        }
        let writers = spaces
            .iter()
            .map(|space| space.quiescence.writer())
            .collect::<Result<Vec<_>, _>>()?;
        let mut transitions = Vec::with_capacity(spaces.len());
        for (space, writer) in spaces.iter().zip(writers) {
            transitions.push(space.quiescence.begin_tracking(writer, interrupt_native)?);
        }
        let closed = self.physical_access().close(&extents);
        for transition in &transitions {
            transition.wait_quiescent();
        }
        closed.wait();
        let result = operation(
            &ProtectionBatch {
                spaces: &spaces,
                aliases,
            },
            &extents,
        );
        match result {
            Ok(value) => {
                for transition in transitions {
                    transition.publish();
                }
                Ok(value)
            }
            Err(error) => {
                let error = Arc::new(error);
                for transition in &transitions {
                    transition.fail(MemoryError::TransitionFailed(Arc::clone(&error)));
                }
                Err(MemoryError::TransitionFailed(error))
            }
        }
    }
    /// Apply the operation directly after the graphics owner has resolved its
    /// exact authority and retained its interval dependency. No endpoint reentry.
    pub fn apply_resolved_access(
        &self,
        ranges: &[AllocationRange],
        operation: &crate::AccessOperation,
    ) -> Result<crate::AccessResult, MemoryError> {
        match operation {
            crate::AccessOperation::ReadNative => self.read_native_resolved(ranges, None),
            crate::AccessOperation::ReadCode(observation)
            | crate::AccessOperation::ReadObserved(observation) => {
                let bytes = self
                    .read_observed_impl(
                        ranges,
                        observation,
                        false,
                        matches!(operation, crate::AccessOperation::ReadCode(_)),
                    )?
                    .expect("owner-resolved code capture cannot defer");
                Ok(crate::AccessResult::Read(bytes))
            }
            crate::AccessOperation::Read => {
                let mut bytes = vec![0; self.validate_ranges(ranges)?];
                self.capture_stream(ranges, &mut bytes)?;
                Ok(crate::AccessResult::Read(bytes.into_boxed_slice()))
            }
            crate::AccessOperation::Write(bytes) => {
                self.write_resolved(ranges, bytes)?;
                Ok(crate::AccessResult::Written)
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
        if let Some(result) = self.try_access_captured(ranges, &operation)? {
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
        operation: &crate::AccessOperation,
    ) -> Result<Option<crate::AccessResult>, MemoryError> {
        let size = self.validate_ranges(ranges)?;
        if ranges.is_empty() {
            return Err(MemoryError::Invalid(
                "captured access requires physical spans",
            ));
        }
        match operation {
            crate::AccessOperation::ReadNative => {
                let mut destination = vec![0; size];
                if self.read_physical_into(ranges, &mut destination)? {
                    return Ok(Some(crate::AccessResult::NativeRead {
                        bytes: destination.into_boxed_slice(),
                        area: None,
                    }));
                }
            }
            crate::AccessOperation::ReadCode(observation)
            | crate::AccessOperation::ReadObserved(observation) => {
                if let Some(bytes) = self.read_observed_impl(
                    ranges,
                    observation,
                    true,
                    matches!(operation, crate::AccessOperation::ReadCode(_)),
                )? {
                    return Ok(Some(crate::AccessResult::Read(bytes)));
                }
            }
            crate::AccessOperation::Write(bytes) => {
                if self.write_physical_impl(ranges, bytes, true)? {
                    return Ok(Some(crate::AccessResult::Written));
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
            crate::AccessOperation::Read => {
                let mut destination = vec![0; size];
                if self.read_physical_into(ranges, &mut destination)? {
                    return Ok(Some(crate::AccessResult::Read(
                        destination.into_boxed_slice(),
                    )));
                }
            }
        }
        Ok(None)
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
                .request_device_access(ranges.capture(offset, bytes)?, crate::AccessOperation::Read)
                .map(crate::Access::Pending);
        }
        Ok(crate::Access::Ready(()))
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
    /// Return a contiguous RAM source for consumption on the graphics owner.
    /// The retained range must outlive every raw load through this pointer.
    /// Subsequent owner effects can change coherence, just as for a native
    /// command span; this is neither a byte snapshot nor a CPU visibility grant.
    pub fn try_ram_pointer(
        &self,
        range: &AllocationRange,
    ) -> Result<Option<crate::HostAddress>, MemoryError> {
        self.validate_ranges(std::slice::from_ref(range))?;
        let mut extents = range.physical_extents();
        let Some(extent) = extents.next() else {
            return Ok(None);
        };
        if extents.next().is_some() {
            return Ok(None);
        }
        let _access = self.physical_access().enter(extent);
        if self.tracking().device.any(extent) {
            return Ok(None);
        }
        Ok(self.arena().host_address(extent.start, extent.bytes))
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
        if destination.is_empty() {
            return Ok(true);
        }
        let _access = self.physical_access().enter(extent);
        if pages(extent).any(|page| self.tracking().device.get(page)) {
            return Ok(false);
        }
        unsafe {
            crate::raw::capture(source.as_ptr(), destination);
        }
        #[cfg(any(test, feature = "performance-counters"))]
        self.tracking()
            .captured_bytes
            .fetch_add(extent.bytes, Ordering::Relaxed);
        Ok(true)
    }
    /// Recover a captured code fetch after the C++ compilation/execution frame
    /// and native admission have ended. Registration precedes the byte copy.
    pub fn access_code_captured(
        &self,
        ranges: Vec<AllocationRange>,
    ) -> Result<(crate::Access<crate::AccessResult>, Observation), MemoryError> {
        let observation = Observation(Arc::new(AtomicBool::new(false)));
        let access = self.access_captured(
            &ranges,
            crate::AccessOperation::ReadCode(observation.clone()),
        )?;
        Ok((access, observation))
    }
    /// The graphics owner supplies only bytes already coherent in RAM. Its
    /// authority determines the reusable interval; this method tracks that
    /// interval while copying only the CPU instruction's requested bytes.
    pub fn read_native_resolved(
        &self,
        ranges: &[AllocationRange],
        reusable: Option<&AllocationRange>,
    ) -> Result<crate::AccessResult, MemoryError> {
        let size = self.validate_ranges(ranges)?;
        let Some(reusable) = reusable else {
            let mut bytes = vec![0; size];
            self.capture_stream(ranges, &mut bytes)?;
            return Ok(crate::AccessResult::NativeRead {
                bytes: bytes.into_boxed_slice(),
                area: None,
            });
        };
        self.validate_ranges(std::slice::from_ref(reusable))?;
        let mut extents = reusable.physical_extents();
        let extent = extents
            .next()
            .ok_or(MemoryError::Invalid("empty native read area"))?;
        if extents.next().is_some()
            || extent.start.get() / PAGE_SIZE != (extent.start.get() + extent.bytes - 1) / PAGE_SIZE
            || ranges.iter().any(|range| {
                range.allocation().identity() != reusable.allocation().identity()
                    || range.physical_extents().any(|part| {
                        part.start < extent.start
                            || part.start.get() + part.bytes > extent.start.get() + extent.bytes
                    })
            })
        {
            return Err(MemoryError::Invalid(
                "native read exceeds its coherent physical area",
            ));
        }
        // The proof concerns device coherence, not an immutable value. CPU
        // stores remain legal and future native reads load their current bytes.
        // Only a native overwrite or physical retirement invalidates this area.
        let observation = Observation::default();
        let _access = self.physical_access().enter(extent);
        self.tracking()
            .register(std::iter::once(extent), &observation, false, true);
        let mut bytes = vec![0; size];
        let mut offset = 0;
        for range in ranges {
            let length = range.bytes() as usize;
            self.capture_unchecked(range, &mut bytes[offset..offset + length]);
            offset += length;
        }
        Ok(crate::AccessResult::NativeRead {
            bytes: bytes.into_boxed_slice(),
            area: Some(crate::NativeReadArea {
                machine: self.identity(),
                allocation: reusable.allocation().identity(),
                start: extent.start,
                bytes: extent.bytes,
                observation,
            }),
        })
    }
    fn read_observed_impl(
        &self,
        ranges: &[AllocationRange],
        observation: &Observation,
        route_device: bool,
        code: bool,
    ) -> Result<Option<Box<[u8]>>, MemoryError> {
        let size = self.validate_ranges(ranges)?;
        if size == 0 {
            return Err(MemoryError::Invalid("code capture requires physical input"));
        }
        let extents: smallvec::SmallVec<[_; 4]> = ranges
            .iter()
            .flat_map(AllocationRange::physical_extents)
            .collect();
        {
            let _access = self.physical_access().enter_many(&extents);
            if route_device
                && extents
                    .iter()
                    .any(|&extent| self.tracking().device.any(extent))
            {
                return Ok(None);
            }
            if extents.iter().all(|&extent| {
                pages(extent).all(|page| {
                    self.tracking().observed(page) && (!code || self.tracking().code(page))
                })
            }) {
                // Existing protection excludes native stores; the admitted set
                // keeps every fragment protected through registration and copy.
                self.tracking()
                    .register(extents.iter().copied(), observation, true, true);
                let mut bytes = vec![0; size];
                let mut offset = 0;
                for range in ranges {
                    let length = range.bytes() as usize;
                    self.capture_unchecked(range, &mut bytes[offset..offset + length]);
                    offset += length;
                }
                return Ok(Some(bytes.into_boxed_slice()));
            }
        }
        self.tracking_transition_with_native(ranges, code, |batch, extents| {
            if route_device
                && extents
                    .iter()
                    .any(|&extent| self.tracking().device.any(extent))
            {
                return Ok(None);
            }
            for &extent in extents {
                self.tracking().observed.update_range(extent, true, |_| {});
                if code {
                    self.tracking().code.update_range(extent, true, |_| {});
                }
            }
            batch.refresh()?;
            let mut bytes = vec![0; size];
            let mut offset = 0;
            for range in ranges {
                self.tracking()
                    .register(range.physical_extents(), observation, true, true);
                let length = range.bytes() as usize;
                self.capture_unchecked(range, &mut bytes[offset..offset + length]);
                offset += length;
            }
            Ok(Some(bytes.into_boxed_slice()))
        })
    }
    /// Gather native input once and observe subsequent CPU writes. Native GPU
    /// coherency belongs to the resource authority; its own device writes do
    /// not invalidate this CPU modification proof. Layout fragments share one
    /// observation and retain no individual pins.
    pub fn capture_clean_scattered(
        &self,
        inputs: &[AllocationRange],
        copies: &[PhysicalCapture],
        destination: &mut [u8],
    ) -> Result<Observation, MemoryError> {
        self.validate_ranges(inputs)?;
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
        for copy in copies {
            let index = retained.partition_point(|extent| extent.start <= copy.source.start);
            if copy.source.bytes == 0
                || index == 0
                || copy
                    .source
                    .start
                    .get()
                    .checked_add(copy.source.bytes)
                    .is_none_or(|end| {
                        end > retained[index - 1].start.get() + retained[index - 1].bytes
                    })
                || copy
                    .destination
                    .checked_add(copy.source.bytes as usize)
                    .is_none_or(|end| end > destination.len())
            {
                return Err(MemoryError::Invalid(
                    "physical capture exceeds retained input or destination",
                ));
            }
        }
        let observation = Observation::default();
        let capture = |destination: &mut [u8]| {
            self.tracking()
                .register(retained.iter().copied(), &observation, true, false);
            for copy in copies {
                let source = self
                    .arena()
                    .host_address(copy.source.start, copy.source.bytes)
                    .expect("validated physical capture");
                let end = copy.destination + copy.source.bytes as usize;
                // Physical exclusion and existing/new write protection keep
                // every writer outside this copy. The destination is borrowed
                // only for the duration of this capture.
                unsafe {
                    crate::raw::capture(source.as_ptr(), &mut destination[copy.destination..end]);
                }
                #[cfg(any(test, feature = "performance-counters"))]
                self.tracking()
                    .captured_bytes
                    .fetch_add(copy.source.bytes, Ordering::Relaxed);
            }
        };
        {
            let _access = self.physical_access().enter_many(&retained);
            if retained
                .iter()
                .all(|&extent| pages(extent).all(|page| self.tracking().observed(page)))
            {
                capture(destination);
                return Ok(observation);
            }
        }
        self.tracking_transition(inputs, |batch, _| {
            for &extent in &retained {
                self.tracking().observed.update_range(extent, true, |_| {});
            }
            batch.refresh()?;
            capture(destination);
            Ok(())
        })?;
        Ok(observation)
    }
    /// Streaming captures keep dirty RAM dirty and do not install observers or
    /// change native protection. Callers supply the owned upload destination.
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
        let mut offset = 0;
        for range in ranges {
            for extent in range.physical_extents() {
                let _access = self.physical_access().enter(extent);
                let bytes = extent.bytes as usize;
                let source = self
                    .arena()
                    .host_address(extent.start, extent.bytes)
                    .expect("retained physical extent");
                unsafe {
                    crate::raw::capture(source.as_ptr(), &mut destination[offset..offset + bytes]);
                }
                offset += bytes;
            }
        }
        #[cfg(any(test, feature = "performance-counters"))]
        self.tracking()
            .captured_bytes
            .fetch_add(destination.len() as u64, Ordering::Relaxed);
        Ok(())
    }
    // Caller either holds physical admission with established write protection,
    // or has closed physical access and established protection on every alias.
    fn capture_unchecked(&self, range: &AllocationRange, destination: &mut [u8]) {
        let mut offset = 0;
        for extent in range.physical_extents() {
            let bytes = extent.bytes as usize;
            let source = self
                .arena()
                .host_address(extent.start, extent.bytes)
                .unwrap();
            unsafe {
                crate::raw::capture(source.as_ptr(), &mut destination[offset..offset + bytes]);
            }
            offset += bytes;
        }
        #[cfg(any(test, feature = "performance-counters"))]
        self.tracking()
            .captured_bytes
            .fetch_add(destination.len() as u64, Ordering::Relaxed);
    }
    /// Revalidate a completed device download while its physical destination is
    /// closed to native stores. Selection runs after draining existing accesses;
    /// only selected byte intervals are copied. No collector lock spans it.
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
        let apply = |batch: Option<&ProtectionBatch<'_>>| {
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
                        if let Some(batch) = batch {
                            batch.invalidate(self, &[target]);
                        }
                        // The transition retains aliases and closes physical access.
                        unsafe {
                            self.write_admitted(target, &source[begin..stop]);
                        }
                    }
                }
                base = end;
            }
            batch.map_or(Ok(()), ProtectionBatch::refresh)
        };
        // Existing device protection already routes native stores through
        // physical admission. Reusing it needs neither alias quiescence nor
        // mprotect; serialize only with mapping/protection changes here.
        let index = self.aliases();
        if extents.iter().all(|&extent| {
            pages(extent).all(|page| self.tracking().device.get(page))
                && !self.tracking().observed.any(extent)
        }) {
            let closed = self.physical_access().close(&extents);
            closed.wait();
            return apply(None);
        }
        drop(index);
        self.tracking_transition(std::slice::from_ref(range), |batch, _| apply(Some(batch)))
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
        self.validate_capture(ranges)?;
        let extents: smallvec::SmallVec<[_; 4]> = ranges
            .physical_extents(offset, source.len() as u64)?
            .collect();
        if source.is_empty() {
            return Ok(crate::Access::Ready(()));
        }
        let written = self.try_write_extents(&extents, source, true);
        if written == Some(true) {
            return Ok(crate::Access::Ready(()));
        }
        // Tracking transitions need exact coordinates, but the parent already
        // owns their lifetime. Only an escaping request acquires independent pins.
        if written.is_none() {
            let selected = ranges.views(offset, source.len() as u64)?;
            if self.write_physical_impl(&selected, source, true)? {
                return Ok(crate::Access::Ready(()));
            }
        }
        self.request_device_access(
            ranges.capture(offset, source.len() as u64)?,
            crate::AccessOperation::Write(source.into()),
        )
        .map(crate::Access::Pending)
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
        if let Some(written) = self.try_write_extents(&extents, source, route_device) {
            return Ok(written);
        }
        self.write_tracked(
            ranges.iter().flat_map(|range| {
                range
                    .physical_extents()
                    .map(|extent| (range.allocation().identity(), extent))
            }),
            source,
            route_device,
        )
    }
    fn write_tracked(
        &self,
        input: impl Iterator<Item = (crate::AllocationId, Extent)>,
        source: &[u8],
        route_device: bool,
    ) -> Result<bool, MemoryError> {
        self.tracking_transition_physical(input, false, |batch, extents| {
            if route_device
                && extents
                    .iter()
                    .any(|&extent| pages(extent).any(|page| self.tracking().device.get(page)))
            {
                return Ok(false);
            }
            batch.invalidate(self, extents);
            let mut offset = 0;
            for &extent in extents {
                let bytes = extent.bytes as usize;
                unsafe {
                    self.write_admitted(extent, &source[offset..offset + bytes]);
                }
                offset += bytes;
            }
            batch.refresh()?;
            Ok(true)
        })
    }
    /// Inputs and payload have already been validated and retained. None needs
    /// an observation transition; Some(false) needs device-owner resolution.
    fn try_write_extents(
        &self,
        extents: &[Extent],
        source: &[u8],
        route_device: bool,
    ) -> Option<bool> {
        let _access = self.physical_access().enter_many(extents);
        if route_device
            && extents
                .iter()
                .any(|&extent| self.tracking().device.any(extent))
        {
            return Some(false);
        }
        if extents
            .iter()
            .any(|&extent| self.tracking().observed.any(extent))
        {
            return None;
        }
        let mut offset = 0;
        for &extent in extents {
            let end = offset + extent.bytes as usize;
            // SAFETY: all destinations remain admitted until the final store.
            unsafe {
                self.write_admitted(extent, &source[offset..end]);
            }
            offset = end;
        }
        Some(true)
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
                .request_device_access(vec![range], crate::AccessOperation::Read)
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
        let written = match machine.try_write_extents(&extents, source, true) {
            Some(written) => written,
            None => machine.write_tracked(
                extents.iter().map(|&extent| (self.identity(), extent)),
                source,
                true,
            )?,
        };
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

#[cfg(test)]
mod observer_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressWidth, GuestVirtualAddress, MemoryPermissions};
    use std::{thread, time::Duration};

    #[test]
    fn accepted_subrange_access_preserves_fragments_observations_and_remapped_bytes() {
        let machine = Machine::new(8 * PAGE_SIZE).unwrap();
        let account = machine.account(8 * PAGE_SIZE).unwrap();
        let mut occupied: Vec<_> = (0..8)
            .map(|_| Some(account.allocate(PAGE_SIZE).unwrap()))
            .collect();
        for index in [1, 3, 5] {
            occupied[index] = None;
        }
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation,
                0,
                3 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let ranges = space
            .capture(base, 3 * PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        assert_eq!(ranges[0].physical_extents().count(), 3);
        let watch = crate::Observation::default();
        machine
            .access_captured(&ranges, crate::AccessOperation::ReadObserved(watch.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        let ranges = crate::CapturedMemory::new(ranges).unwrap();
        space.unmap(base, 3 * PAGE_SIZE).unwrap();
        let replacement = occupied[0].as_ref().unwrap().clone();
        space
            .map(
                base,
                replacement,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let input = [17; 12];
        assert!(matches!(
            machine
                .write_captured(&ranges, PAGE_SIZE - 4, &input)
                .unwrap(),
            crate::Access::Ready(())
        ));
        assert!(watch.is_invalid());
        let mut output = [0; 16];
        assert!(matches!(
            machine
                .read_captured_into(&ranges, PAGE_SIZE - 6, &mut output)
                .unwrap(),
            crate::Access::Ready(())
        ));
        assert_eq!(&output[..2], &[0; 2]);
        assert_eq!(&output[2..14], &input);
        assert_eq!(&output[14..], &[0; 2]);
        assert!(
            machine
                .write_captured(&ranges, 3 * PAGE_SIZE - 1, &[99; 2])
                .is_err()
        );
        assert!(matches!(
            space.read_value(base, 8).unwrap(),
            crate::Access::Ready(0)
        ));
    }

    #[test]
    fn native_ram_observations_survive_cpu_changes_but_not_device_writes_or_retirement() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let field = AllocationRange::new(allocation.pin(), 128, 4).unwrap();
        let reusable = AllocationRange::new(allocation.pin(), 64, 448).unwrap();
        let ordinary = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&field),
                crate::AccessOperation::ReadObserved(ordinary.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let crate::AccessResult::NativeRead {
            area: Some(area), ..
        } = machine
            .read_native_resolved(std::slice::from_ref(&field), Some(&reusable))
            .unwrap()
        else {
            panic!("expected native read area");
        };
        machine
            .write_resolved(std::slice::from_ref(&field), &9u32.to_le_bytes())
            .unwrap();
        assert!(ordinary.is_invalid());
        assert!(!area.observation().is_invalid());
        machine
            .publish_device_write(std::slice::from_ref(&field), || Ok(()))
            .unwrap();
        assert!(area.observation().is_invalid());
        let crate::AccessResult::NativeRead {
            area: Some(after_device),
            ..
        } = machine
            .read_native_resolved(std::slice::from_ref(&field), Some(&reusable))
            .unwrap()
        else {
            panic!("expected renewed native read area");
        };
        assert!(!after_device.observation().is_invalid());
        drop(field);
        drop(reusable);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(after_device.observation().is_invalid());
        assert_eq!(machine.tracking_statistics().observer_registrations, 0);
    }

    #[test]
    fn native_read_area_cannot_authorize_another_machine_with_the_same_allocation_id() {
        let mut machines = Vec::new();
        for value in [17u32, 29] {
            let machine = Machine::new(PAGE_SIZE).unwrap();
            let allocation = machine
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap();
            let field = AllocationRange::new(allocation.pin(), 128, 4).unwrap();
            machine
                .write_resolved(std::slice::from_ref(&field), &value.to_le_bytes())
                .unwrap();
            let space = mapping(&machine, allocation, PAGE_SIZE);
            machine
                .publish_device_write(std::slice::from_ref(&field), || Ok(()))
                .unwrap();
            let crate::AccessResult::NativeRead {
                area: Some(area), ..
            } = machine
                .read_native_resolved(std::slice::from_ref(&field), Some(&field))
                .unwrap()
            else {
                panic!("expected native read area");
            };
            machines.push((space, area));
        }
        let (space, area) = &machines[1];
        assert_eq!(area.allocation, machines[0].1.allocation);
        assert_eq!(area.start, machines[0].1.start);
        assert!(!machines[0].1.observation().is_invalid());
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        let address = GuestVirtualAddress::new(PAGE_SIZE + 128);
        assert_eq!(
            space
                .read_native(&native, address, 4, Some(&machines[0].1))
                .unwrap(),
            None
        );
        assert_eq!(
            space.read_native(&native, address, 4, Some(area)).unwrap(),
            Some(29)
        );
    }

    fn mapping(machine: &Machine, allocation: crate::Allocation, base: u64) -> Arc<AddressSpace> {
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let bytes = allocation.bytes();
        space
            .map(
                GuestVirtualAddress::new(base),
                allocation,
                0,
                bytes,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
    }
    fn host_store(space: &AddressSpace, address: u64) -> bool {
        let pointer = space
            .native_data_base()
            .as_ptr()
            .wrapping_add(address as usize);
        let host = pointer as usize;
        // Inspect the kernel's actual VM permissions. A denied write is never
        // attempted in-process, so the test needs no signal handler or ptrace.
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        for line in maps.lines() {
            let mut fields = line.split_whitespace();
            let (start, end) = fields.next().unwrap().split_once('-').unwrap();
            let start = usize::from_str_radix(start, 16).unwrap();
            let end = usize::from_str_radix(end, 16).unwrap();
            if (start..end).contains(&host) {
                if fields.next().unwrap().as_bytes()[1] != b'w' {
                    return false;
                }
                // This test owns the space and is outside every transition.
                unsafe {
                    crate::raw::write_value(pointer, 1, 42);
                }
                return true;
            }
        }
        panic!("live virtual arena absent from host mappings")
    }

    #[test]
    fn first_data_capture_protects_aliases_without_interrupting_native_execution() {
        use std::sync::mpsc;
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
        let space = mapping(&machine, allocation, PAGE_SIZE);
        let interrupts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let interrupted = Arc::clone(&interrupts);
        let running = Arc::clone(&space);
        let (entered, active) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let native = thread::spawn(move || {
            let core = running.register_core(Arc::new(move || {
                interrupted.fetch_add(1, Ordering::Relaxed);
            }));
            let _native = core.enter_native().unwrap();
            entered.send(()).unwrap();
            released.recv().unwrap();
        });
        active.recv().unwrap();
        let read = machine.clone();
        let (completed, result) = mpsc::channel();
        let reader = thread::spawn(move || {
            let observation = Observation::default();
            let captured = read
                .access_captured(
                    &[range],
                    crate::AccessOperation::ReadObserved(observation.clone()),
                )
                .map(|access| {
                    access.into_ready().unwrap();
                    observation
                });
            completed.send(captured).unwrap();
        });
        let captured_while_running = result.recv_timeout(Duration::from_secs(1));
        release.send(()).unwrap();
        native.join().unwrap();
        reader.join().unwrap();
        let observation = captured_while_running
            .expect("data tracking interrupted native execution")
            .unwrap();
        assert_eq!(interrupts.load(Ordering::Relaxed), 0);
        assert!(!host_store(&space, PAGE_SIZE));
        assert!(!observation.is_invalid());
        space
            .write_value(GuestVirtualAddress::new(PAGE_SIZE), 4, 7)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(observation.is_invalid());
        assert!(host_store(&space, PAGE_SIZE));
    }

    #[test]
    fn another_capture_of_protected_ram_neither_interrupts_a_running_core_nor_reprotects() {
        use crate::{Access, AccessOperation, AccessResult};
        use std::sync::mpsc;
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
        let space = mapping(&machine, allocation, PAGE_SIZE);
        let first = Observation::default();
        assert!(matches!(
            machine
                .access_captured(
                    &[range.slice(0, 4).unwrap()],
                    AccessOperation::ReadObserved(first.clone())
                )
                .unwrap(),
            Access::Ready(AccessResult::Read(_))
        ));
        let before = machine.tracking_statistics().protection_calls;
        let interrupts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let interrupted = Arc::clone(&interrupts);
        let running = Arc::clone(&space);
        let (entered, active) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let native = thread::spawn(move || {
            let core = running.register_core(Arc::new(move || {
                interrupted.fetch_add(1, Ordering::Relaxed);
            }));
            let _native = core.enter_native().unwrap();
            entered.send(()).unwrap();
            released.recv().unwrap();
        });
        active.recv().unwrap();
        let read = machine.clone();
        let (completed, result) = mpsc::channel();
        let reader = thread::spawn(move || {
            let observation = Observation::default();
            let result = read
                .access_captured(
                    &[range.slice(4, 4).unwrap()],
                    AccessOperation::ReadObserved(observation.clone()),
                )
                .unwrap();
            let mut bytes = [0xaa; 12];
            let scattered = read
                .capture_clean_scattered(
                    std::slice::from_ref(&range),
                    &[PhysicalCapture {
                        source: Extent {
                            start: range.physical_extents().next().unwrap().start,
                            bytes: 8,
                        },
                        destination: 2,
                    }],
                    &mut bytes,
                )
                .unwrap();
            completed
                .send((result, observation, scattered, bytes))
                .unwrap();
        });
        let captured_while_running = result.recv_timeout(Duration::from_secs(1));
        release.send(()).unwrap();
        native.join().unwrap();
        reader.join().unwrap();
        let (captured, second, scattered, bytes) =
            captured_while_running.expect("protected RAM capture must not drain the running core");
        assert!(matches!(captured, Access::Ready(AccessResult::Read(_))));
        assert_eq!(bytes, [0xaa, 0xaa, 0, 0, 0, 0, 0, 0, 0, 0, 0xaa, 0xaa]);
        assert_eq!(interrupts.load(Ordering::Relaxed), 0);
        assert_eq!(machine.tracking_statistics().protection_calls, before);
        assert!(matches!(
            space
                .write_value(GuestVirtualAddress::new(PAGE_SIZE + 4), 4, 7)
                .unwrap(),
            Access::Ready(())
        ));
        assert!(first.is_invalid());
        assert!(second.is_invalid());
        assert!(scattered.is_invalid());
    }
    #[test]
    fn fragmented_ram_writes_do_not_publish_dirty_pages_and_defer_device_writes_intact() {
        let machine = Machine::new(6 * PAGE_SIZE).unwrap();
        let account = machine.account(6 * PAGE_SIZE).unwrap();
        let mut occupied: Vec<_> = (0..6)
            .map(|_| Some(account.allocate(PAGE_SIZE).unwrap()))
            .collect();
        for index in [0, 2, 4] {
            occupied[index] = None;
        }
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let full = AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap();
        assert_eq!(full.physical_extents().count(), 3);
        let _space = mapping(&machine, allocation, PAGE_SIZE);
        let split = [
            full.slice(0, PAGE_SIZE).unwrap(),
            full.slice(PAGE_SIZE, 2 * PAGE_SIZE).unwrap(),
        ];
        machine.take_cpu_modified_pages();
        let protections = machine.tracking_statistics().protection_calls;
        for ranges in [std::slice::from_ref(&full), split.as_slice()] {
            machine
                .write_resolved(ranges, &vec![3; 3 * PAGE_SIZE as usize])
                .unwrap();
            assert!(machine.take_cpu_modified_pages().is_empty());
            let ranges = crate::CapturedMemory::new(ranges.iter().cloned()).unwrap();
            machine
                .write_captured(&ranges, PAGE_SIZE - 2, &[7; 8])
                .unwrap()
                .into_ready()
                .unwrap();
            assert!(machine.take_cpu_modified_pages().is_empty());
            let mut bytes = [0; 12];
            machine
                .read_captured_into(&ranges, PAGE_SIZE - 4, &mut bytes)
                .unwrap()
                .into_ready()
                .unwrap();
            assert_eq!(bytes, [3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3]);
        }
        assert_eq!(machine.tracking_statistics().protection_calls, protections);
        machine
            .publish_device_write(&[full.slice(PAGE_SIZE, PAGE_SIZE).unwrap()], || Ok(()))
            .unwrap();
        assert!(
            machine
                .try_access_captured(
                    &split,
                    &crate::AccessOperation::Write(vec![9; 3 * PAGE_SIZE as usize].into())
                )
                .unwrap()
                .is_none()
        );
        let mut prefix = [0; 4];
        machine
            .capture_stream(&[full.slice(0, 4).unwrap()], &mut prefix)
            .unwrap();
        assert_eq!(prefix, [3; 4]);
    }

    #[test]
    fn fragmented_observations_reuse_code_protection_without_stopping_native_execution() {
        use crate::{AccessOperation, AccessResult};
        use std::sync::{atomic::AtomicUsize, mpsc};
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(4 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let full = AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap();
        let space = mapping(&machine, allocation, PAGE_SIZE);
        let ranges = [
            full.slice(0, 8).unwrap(),
            full.slice(2 * PAGE_SIZE, 8).unwrap(),
        ];
        let first = Observation::default();
        machine
            .access_captured(&ranges, AccessOperation::ReadCode(first.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        let protections = machine.tracking_statistics().protection_calls;
        let interrupts = Arc::new(AtomicUsize::new(0));
        let interrupted = interrupts.clone();
        let running = space.clone();
        let (entered, active) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let native = thread::spawn(move || {
            let core = running.register_core(Arc::new(move || {
                interrupted.fetch_add(1, Ordering::Relaxed);
            }));
            let _native = core.enter_native().unwrap();
            entered.send(()).unwrap();
            released.recv().unwrap();
        });
        active.recv().unwrap();
        let read = machine.clone();
        let (completed, result) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut observations = Vec::new();
            for code in [false, true] {
                let observation = Observation::default();
                let operation = if code {
                    AccessOperation::ReadCode(observation.clone())
                } else {
                    AccessOperation::ReadObserved(observation.clone())
                };
                let AccessResult::Read(bytes) = read
                    .access_captured(&ranges, operation)
                    .unwrap()
                    .into_ready()
                    .unwrap()
                else {
                    panic!("expected read");
                };
                assert_eq!(&*bytes, &[0; 16]);
                observations.push(observation);
            }
            let copies: Vec<_> = ranges
                .iter()
                .enumerate()
                .map(|(index, range)| PhysicalCapture {
                    source: range.physical_extents().next().unwrap(),
                    destination: index * 8,
                })
                .collect();
            let mut bytes = [0xaa; 20];
            observations.push(
                read.capture_clean_scattered(&ranges, &copies, &mut bytes)
                    .unwrap(),
            );
            assert_eq!(&bytes[..16], &[0; 16]);
            assert_eq!(&bytes[16..], &[0xaa; 4]);
            completed.send(observations).unwrap();
        });
        let captured = result.recv_timeout(Duration::from_secs(2));
        release.send(()).unwrap();
        native.join().unwrap();
        reader.join().unwrap();
        let observations =
            captured.expect("existing fragmented protection must not drain native execution");
        assert_eq!(interrupts.load(Ordering::Relaxed), 0);
        assert_eq!(machine.tracking_statistics().protection_calls, protections);
        assert!(observations.iter().all(|o| !o.is_invalid()));
        space
            .write_value(GuestVirtualAddress::new(3 * PAGE_SIZE), 4, 7)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(first.is_invalid());
        assert!(observations.iter().all(Observation::is_invalid));
    }

    #[test]
    fn fragmented_capture_establishes_missing_tracking_and_keeps_alias_invalidation() {
        use crate::AccessOperation;
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let full = AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap();
        let first_alias = mapping(&machine, allocation.clone(), PAGE_SIZE);
        let other_alias = mapping(&machine, allocation, 8 * PAGE_SIZE);
        let ranges = [
            full.slice(0, 8).unwrap(),
            full.slice(2 * PAGE_SIZE, 8).unwrap(),
        ];
        let first = Observation::default();
        machine
            .access_captured(&ranges[..1], AccessOperation::ReadObserved(first.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        let code = Observation::default();
        machine
            .access_captured(&ranges, AccessOperation::ReadCode(code.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        for range in &ranges {
            let page = range.physical_extents().next().unwrap().start.get() / PAGE_SIZE;
            assert!(machine.tracking().observed(page));
            assert!(machine.tracking().code(page));
        }
        let scattered = machine
            .capture_clean_scattered(
                &ranges,
                &[
                    PhysicalCapture {
                        source: ranges[0].physical_extents().next().unwrap(),
                        destination: 0,
                    },
                    PhysicalCapture {
                        source: ranges[1].physical_extents().next().unwrap(),
                        destination: 8,
                    },
                ],
                &mut [0; 16],
            )
            .unwrap();
        other_alias
            .write_value(GuestVirtualAddress::new(10 * PAGE_SIZE), 4, 91)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(code.is_invalid());
        assert!(scattered.is_invalid());
        assert!(!first.is_invalid());
        let mut value = [0; 4];
        first_alias
            .read_into(GuestVirtualAddress::new(3 * PAGE_SIZE), &mut value)
            .unwrap()
            .into_ready()
            .unwrap();
        assert_eq!(value, 91u32.to_le_bytes());
        first_alias
            .write_value(GuestVirtualAddress::new(PAGE_SIZE), 4, 12)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(first.is_invalid());
    }

    fn invalidations(context: &crate::CoreContext) -> Vec<crate::CodeInvalidation> {
        let native = context.enter_native().unwrap();
        let mut ranges = Vec::new();
        native
            .synchronize_code(|range| {
                ranges.push(range);
                Ok(())
            })
            .unwrap();
        ranges
    }
    #[test]
    fn code_write_quiesces_every_executable_alias_before_publishing_new_instructions() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let data = mapping(&machine, allocation.clone(), 0x20000);
        let mut executable = Vec::new();
        for base in [0x1000, 0x9000] {
            let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
            space
                .map(
                    GuestVirtualAddress::new(base),
                    allocation.clone(),
                    0,
                    2 * PAGE_SIZE,
                    MemoryPermissions::READ_EXECUTE,
                )
                .unwrap();
            executable.push((space, base));
        }
        let spans = data
            .capture(
                GuestVirtualAddress::new(0x20010),
                4,
                MemoryPermissions::READ,
            )
            .unwrap();
        let code = crate::Observation::default();
        let crate::AccessResult::Read(code_bytes) = machine
            .access_captured(&spans, crate::AccessOperation::ReadCode(code.clone()))
            .unwrap()
            .into_ready()
            .unwrap()
        else {
            panic!("expected captured bytes")
        };
        assert_eq!(&*code_bytes, &[0; 4]);
        assert!(!host_store(&data, 0x20010));
        let data_context = data.register_core(Arc::new(|| {}));
        let (interrupt, interrupts) = std::sync::mpsc::channel();
        let (ready, running) = std::sync::mpsc::channel();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut cores = Vec::new();
        for (space, base) in executable {
            let interrupt = interrupt.clone();
            let context = space.register_core(Arc::new(move || {
                interrupt.send(()).unwrap();
            }));
            let ready = ready.clone();
            let barrier = Arc::clone(&barrier);
            cores.push(thread::spawn(move || {
                let native = context.enter_native().unwrap();
                ready.send(()).unwrap();
                barrier.wait();
                native.suspend();
                native.resume().unwrap();
                let mut ranges = Vec::new();
                native
                    .synchronize_code(|range| {
                        ranges.push(range);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(
                    ranges,
                    [crate::CodeInvalidation {
                        start: GuestVirtualAddress::new(base),
                        bytes: PAGE_SIZE,
                    }]
                );
                let pointer = space
                    .native_data_base()
                    .as_ptr()
                    .wrapping_add(base as usize + 16);
                // The admitted core sees only the published instruction bytes.
                assert_eq!(unsafe { crate::raw::read_value(pointer, 4) }, 0xd503201f);
            }));
        }
        running.recv().unwrap();
        running.recv().unwrap();
        let writer_space = Arc::clone(&data);
        let writer = thread::spawn(move || {
            writer_space
                .write_value(GuestVirtualAddress::new(0x20010), 4, 0xd503201f)
                .unwrap()
                .into_ready()
                .unwrap();
        });
        interrupts.recv_timeout(Duration::from_secs(1)).unwrap();
        interrupts.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(!code.is_invalid());
        barrier.wait();
        writer.join().unwrap();
        for core in cores {
            core.join().unwrap();
        }
        assert!(code.is_invalid());
        assert!(invalidations(&data_context).is_empty());
        assert!(host_store(&data, 0x20010));
    }
    #[test]
    fn failed_atomics_preserve_code_and_successful_writers_invalidate_it_once() {
        use crate::{AccessOrdering, AccessResult, AtomicAccess, AtomicOperation};
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x4000);
        space
            .map(base, allocation, 0, PAGE_SIZE, MemoryPermissions::ALL)
            .unwrap();
        let context = space.register_core(Arc::new(|| {}));
        let ranges = space.capture(base, 4, MemoryPermissions::ALL).unwrap();
        let range = &ranges[0];
        let (_, old) = machine.load_exclusive_resolved(range).unwrap();
        machine
            .write_resolved(&ranges, &1u32.to_le_bytes())
            .unwrap();
        for writer in 0..3 {
            let code = crate::Observation::default();
            machine
                .access_captured(&ranges, crate::AccessOperation::ReadCode(code.clone()))
                .unwrap()
                .into_ready()
                .unwrap();
            assert!(!machine.store_exclusive_resolved(range, old, 5).unwrap());
            assert!(matches!(
                machine
                    .atomic_physical(
                        range,
                        AtomicAccess {
                            bytes: 4,
                            ordering: AccessOrdering::AcquireRelease,
                            operation: AtomicOperation::CompareExchange {
                                expected: 999,
                                replacement: 10
                            },
                        }
                    )
                    .unwrap(),
                AccessResult::Atomic { written: false, .. }
            ));
            assert!(!code.is_invalid());
            assert!(invalidations(&context).is_empty());
            match writer {
                0 => {
                    let (value, reservation) = machine.load_exclusive_resolved(range).unwrap();
                    assert!(
                        machine
                            .store_exclusive_resolved(range, reservation, value + 1)
                            .unwrap()
                    );
                }
                1 => {
                    machine
                        .atomic_physical(
                            range,
                            AtomicAccess {
                                bytes: 4,
                                ordering: AccessOrdering::AcquireRelease,
                                operation: AtomicOperation::ReadModifyWrite {
                                    transform: crate::AtomicTransform::Add,
                                    operand: 1,
                                },
                            },
                        )
                        .unwrap();
                }
                2 => machine
                    .write_resolved(&ranges, &4u32.to_le_bytes())
                    .unwrap(),
                _ => unreachable!(),
            }
            assert!(code.is_invalid());
            assert_eq!(
                invalidations(&context),
                [crate::CodeInvalidation {
                    start: base,
                    bytes: PAGE_SIZE
                }]
            );
            machine
                .write_resolved(&ranges, &4u32.to_le_bytes())
                .unwrap();
            assert!(invalidations(&context).is_empty());
        }
    }
    #[test]
    fn native_cpu_tracking_survives_device_writes_but_still_observes_a_later_cpu_store() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
        let physical = input.physical_extents().next().unwrap().start;
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x7000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let mut bytes = [0; 4];
        let native = machine
            .capture_clean_scattered(
                std::slice::from_ref(&input),
                &[PhysicalCapture {
                    source: Extent {
                        start: physical,
                        bytes: 4,
                    },
                    destination: 0,
                }],
                &mut bytes,
            )
            .unwrap();
        let snapshot = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&input),
                crate::AccessOperation::ReadObserved(snapshot.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        machine
            .publish_device_write(std::slice::from_ref(&input), || {
                assert!(snapshot.is_invalid());
                assert!(!native.is_invalid());
                Ok(())
            })
            .unwrap();
        assert!(machine.tracking().observed(physical.get() / PAGE_SIZE));
        assert!(
            !machine
                .tracking()
                .device_observed
                .get(physical.get() / PAGE_SIZE)
        );
        let protected = machine.tracking_statistics().protection_calls;
        machine
            .invalidate_device_observations(std::slice::from_ref(&input), || Ok(()))
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, protected);
        assert!(!native.is_invalid());
        machine
            .retire_device_pages(std::slice::from_ref(&input))
            .unwrap();
        assert!(
            !host_store(&space, base.get()),
            "CPU observation must remain protected after device retirement"
        );
        space.write_value(base, 4, 7).unwrap().into_ready().unwrap();
        assert!(native.is_invalid());
        assert!(host_store(&space, base.get()));
    }
    #[test]
    fn repeated_device_publication_invalidates_new_code_without_reprotecting_pages() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x7000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_EXECUTE,
            )
            .unwrap();
        let context = space.register_core(Arc::new(|| {}));
        let ranges = space.capture(base, 4, MemoryPermissions::EXECUTE).unwrap();
        machine.publish_device_write(&ranges, || Ok(())).unwrap();
        let before = machine.tracking_statistics().protection_calls;
        // The owner has resolved this exact demand; other bytes in the page
        // can remain device-produced, so the device page stays inaccessible.
        let code = crate::Observation::default();
        machine
            .apply_resolved_access(&ranges, &crate::AccessOperation::ReadCode(code.clone()))
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before);
        assert!(invalidations(&context).is_empty());
        machine
            .publish_device_write(&ranges, || {
                assert!(code.is_invalid());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            invalidations(&context),
            [crate::CodeInvalidation {
                start: base,
                bytes: PAGE_SIZE
            }]
        );
        assert_eq!(machine.tracking_statistics().protection_calls, before);
    }
    #[test]
    fn captures_cannot_consume_other_views_pending_cpu_modification_pages() {
        let machine = Machine::new(68 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(68 * PAGE_SIZE)
            .unwrap()
            .allocate(68 * PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 68 * PAGE_SIZE).unwrap();
        let first = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&input),
                crate::AccessOperation::ReadObserved(first.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        machine.take_cpu_modified_pages();
        for offset in [65 * PAGE_SIZE + 7, PAGE_SIZE + 63] {
            let write = AllocationRange::new(allocation.pin(), offset, 4).unwrap();
            machine.write_resolved(&[write], &[7; 4]).unwrap();
        }
        let second = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&input),
                crate::AccessOperation::ReadObserved(second.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(first.is_invalid());
        assert!(!second.is_invalid());
        let mut pages = machine.take_cpu_modified_pages();
        pages.sort_unstable_by_key(|extent| extent.start);
        let start = input.physical_extents().next().unwrap().start.get();
        assert_eq!(
            pages,
            [
                Extent {
                    start: ArenaOffset::new(start + PAGE_SIZE),
                    bytes: PAGE_SIZE
                },
                Extent {
                    start: ArenaOffset::new(start + 65 * PAGE_SIZE),
                    bytes: PAGE_SIZE
                }
            ]
        );
        assert!(machine.take_cpu_modified_pages().is_empty());
        let write = AllocationRange::new(allocation.pin(), PAGE_SIZE + 3, 1).unwrap();
        machine.write_resolved(&[write], &[9]).unwrap();
        assert_eq!(
            machine.take_cpu_modified_pages(),
            [Extent {
                start: ArenaOffset::new(start + PAGE_SIZE),
                bytes: PAGE_SIZE
            }]
        );
    }

    #[test]
    fn first_write_invalidates_all_observers_and_restores_every_physical_alias() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(4 * PAGE_SIZE)
            .unwrap()
            .allocate(4 * PAGE_SIZE)
            .unwrap();
        let first = mapping(&machine, allocation.clone(), 0x4000);
        let second = mapping(&machine, allocation, 0x9000);
        assert_eq!(machine.tracking_statistics().protection_calls, 0);
        assert!(host_store(&first, 0x4000));
        let ranges = first
            .capture(
                GuestVirtualAddress::new(0x4000),
                2 * PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let a = crate::Observation::default();
        machine
            .access_captured(&ranges, crate::AccessOperation::ReadObserved(a.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        let b = crate::Observation::default();
        machine
            .access_captured(&ranges, crate::AccessOperation::ReadObserved(b.clone()))
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!a.is_invalid());
        assert!(!b.is_invalid());
        assert!(!host_store(&first, 0x4000));
        assert!(!host_store(&second, 0x9000));
        let protected = machine.tracking_statistics().protection_calls;
        assert_eq!(protected, 2);
        second
            .write_value(GuestVirtualAddress::new(0x9001), 4, 0x12345678)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(a.is_invalid());
        assert!(b.is_invalid());
        assert!(host_store(&first, 0x4000));
        assert!(host_store(&second, 0x9000));
        // The adjacent observed page remains protected until its own store.
        assert!(!host_store(&second, 0xa000));
        assert_eq!(
            first
                .read_value(GuestVirtualAddress::new(0x4001), 4)
                .unwrap()
                .into_ready()
                .unwrap(),
            0x12345678
        );
        second
            .write(GuestVirtualAddress::new(0xafff), &[1, 2])
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(host_store(&second, 0xa000));
        let before = machine.tracking_statistics();
        for _ in 0..32 {
            second
                .write_value(GuestVirtualAddress::new(0x9001), 4, 19)
                .unwrap()
                .into_ready()
                .unwrap();
        }
        assert_eq!(machine.tracking_statistics(), before);
    }
    #[test]
    fn new_alias_and_guest_permission_replacement_preserve_existing_tracking() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let first = mapping(&machine, allocation.clone(), 0x1000);
        let ranges = first
            .capture(
                GuestVirtualAddress::new(0x1000),
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let observed = crate::Observation::default();
        machine
            .access_captured(
                &ranges,
                crate::AccessOperation::ReadObserved(observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let second = mapping(&machine, allocation, 0x6000);
        assert!(!host_store(&second, 0x6000));
        second
            .protect(
                GuestVirtualAddress::new(0x6000),
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        second
            .protect(
                GuestVirtualAddress::new(0x6000),
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        assert!(!host_store(&second, 0x6000));
        second
            .write(GuestVirtualAddress::new(0x6000), &[5])
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(observed.is_invalid());
        assert_eq!(
            first
                .read_value(GuestVirtualAddress::new(0x1000), 1)
                .unwrap()
                .into_ready()
                .unwrap(),
            5
        );
    }
    #[test]
    fn partial_unmap_replaces_reverse_ranges_and_capture_does_not_pin_freed_ram() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let pin = allocation.pin();
        let cached_identity = pin.downgrade();
        let space = mapping(&machine, allocation, 0x3000);
        space
            .unmap(GuestVirtualAddress::new(0x4000), PAGE_SIZE)
            .unwrap();
        assert_eq!(machine.aliases().index.len(), 2);
        let ranges = vec![AllocationRange::new(pin, 0, 3 * PAGE_SIZE).unwrap()];
        let observed = crate::Observation::default();
        machine
            .access_captured(
                &ranges,
                crate::AccessOperation::ReadObserved(observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!host_store(&space, 0x3000));
        assert!(!host_store(&space, 0x5000));
        drop(space);
        assert!(machine.aliases().index.is_empty());
        assert!(!observed.is_invalid());
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().retained_bytes, 3 * PAGE_SIZE);
        drop(ranges);
        assert!(observed.is_invalid());
        assert!(!cached_identity.is_live());
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(machine.tracking_statistics().observer_pages, 0);
        let recycled = account.allocate(3 * PAGE_SIZE).unwrap();
        let fresh = mapping(&machine, recycled, 0x7000);
        assert!(host_store(&fresh, 0x7000));
    }
    #[test]
    fn code_capture_begins_all_core_interrupts_before_waiting_and_tracks_host_writes() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let first = mapping(&machine, allocation.clone(), 0x1000);
        let second = mapping(&machine, allocation, 0x1000);
        let ranges = first
            .capture(
                GuestVirtualAddress::new(0x1000),
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let (interrupt, interrupts) = std::sync::mpsc::channel();
        let mut cores = Vec::new();
        let (ready, running) = std::sync::mpsc::channel();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        for space in [&first, &second] {
            let interrupt = interrupt.clone();
            let context = space.register_core(Arc::new(move || {
                interrupt.send(()).unwrap();
            }));
            let barrier = Arc::clone(&barrier);
            let ready = ready.clone();
            cores.push(thread::spawn(move || {
                let native = context.enter_native().unwrap();
                ready.send(()).unwrap();
                barrier.wait();
                native.suspend();
                native.resume().unwrap();
            }));
        }
        running.recv().unwrap();
        running.recv().unwrap();
        let capture_machine = machine.clone();
        let capture = thread::spawn(move || {
            let observation = Observation::default();
            capture_machine
                .access_captured(
                    &ranges,
                    crate::AccessOperation::ReadCode(observation.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            observation
        });
        // Both requests must be issued even while the first core remains active.
        interrupts.recv_timeout(Duration::from_secs(1)).unwrap();
        interrupts.recv_timeout(Duration::from_secs(1)).unwrap();
        barrier.wait();
        let observation = capture.join().unwrap();
        for core in cores {
            core.join().unwrap();
        }
        first
            .write(GuestVirtualAddress::new(0x1000), &[17])
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(observation.is_invalid());
    }
    #[test]
    fn dirty_streaming_capture_does_not_install_tracking_or_clear_dirty_summaries() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let ranges = [AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap()];
        let before = machine.tracking_statistics();
        for byte in 0..8 {
            let source = vec![byte; (2 * PAGE_SIZE) as usize];
            machine.write_resolved(&ranges, &source).unwrap();
            let mut destination = vec![0; source.len()];
            machine.capture_stream(&ranges, &mut destination).unwrap();
            assert_eq!(source, destination);
        }
        assert_eq!(
            machine.tracking_statistics().protection_calls,
            before.protection_calls
        );
        assert_eq!(machine.tracking_statistics().observer_pages, 0);
    }
    #[test]
    fn exclusive_cas_covers_alias_scalar_bulk_and_physical_writes() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let first = mapping(&machine, allocation.clone(), 0x1000);
        let second = mapping(&machine, allocation, 0x8000);
        let range = first
            .capture(
                GuestVirtualAddress::new(0x1040),
                8,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap()
            .remove(0);
        for writer in 0..3 {
            machine
                .write_resolved(std::slice::from_ref(&range), &0u64.to_le_bytes())
                .unwrap();
            let (value, reservation) = machine.load_exclusive_resolved(&range).unwrap();
            assert_eq!(value, 0);
            let before = machine.tracking_statistics().protection_calls;
            match writer {
                0 => {
                    second
                        .write_value(GuestVirtualAddress::new(0x8040), 8, 9)
                        .unwrap()
                        .into_ready()
                        .unwrap();
                }
                1 => {
                    second
                        .write(GuestVirtualAddress::new(0x803f), &[0, 9])
                        .unwrap()
                        .into_ready()
                        .unwrap();
                }
                2 => {
                    machine
                        .write_resolved(std::slice::from_ref(&range), &9u64.to_le_bytes())
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                !machine
                    .store_exclusive_resolved(&range, reservation, 5)
                    .unwrap()
            );
            assert_eq!(machine.tracking_statistics().protection_calls, before);
        }
        let (_, reservation) = machine.load_exclusive_resolved(&range).unwrap();
        // A different granule in the same physical page does not invalidate it.
        second
            .write_value(GuestVirtualAddress::new(0x8000), 8, 7)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(
            machine
                .store_exclusive_resolved(&range, reservation, 13)
                .unwrap()
        );
        assert!(
            !machine
                .store_exclusive_resolved(&range, reservation, 15)
                .unwrap()
        );
        assert_eq!(
            first
                .read_value(GuestVirtualAddress::new(0x1040), 8)
                .unwrap()
                .into_ready()
                .unwrap(),
            13
        );
        assert_eq!(machine.tracking_statistics().protection_calls, 0);
    }
    #[test]
    fn observations_follow_exact_physical_lifetime_after_cpu_guest_release() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
        let first = AllocationRange::new(allocation.pin_range(0, PAGE_SIZE).unwrap(), 0, PAGE_SIZE)
            .unwrap();
        let native = AllocationRange::new(
            allocation.pin_range(PAGE_SIZE, PAGE_SIZE).unwrap(),
            PAGE_SIZE,
            PAGE_SIZE,
        )
        .unwrap();
        let observations = [&first, &native].map(|range| {
            let observation = Observation::default();
            machine
                .access_captured(
                    std::slice::from_ref(range),
                    crate::AccessOperation::ReadObserved(observation.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            observation
        });
        drop(first);
        drop(allocation);
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert!(observations[0].is_invalid());
        assert!(!observations[1].is_invalid());
        let captured_after_cpu_release = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&native),
                crate::AccessOperation::ReadObserved(captured_after_cpu_release.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!captured_after_cpu_release.is_invalid());
        machine
            .write_resolved(std::slice::from_ref(&native), &vec![7; PAGE_SIZE as usize])
            .unwrap();
        assert!(observations[1].is_invalid());
        assert!(captured_after_cpu_release.is_invalid());
        let last = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&native),
                crate::AccessOperation::ReadObserved(last.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!last.is_invalid());
        drop(native);
        assert!(last.is_invalid());
        assert_eq!(machine.statistics().allocated_bytes, 0);
        let recycled = account.allocate(PAGE_SIZE).unwrap();
        assert!(last.is_invalid());
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        drop(recycled);
    }
    #[test]
    fn plain_native_value_change_rejects_checked_exclusive_without_invalidating_a_later_observer() {
        for bytes in [1, 2, 4, 8, 16] {
            let machine = Machine::new(PAGE_SIZE).unwrap();
            let allocation = machine
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap();
            let range = AllocationRange::new(allocation.pin(), 0, bytes).unwrap();
            let (_, reservation) = machine.load_exclusive_resolved(&range).unwrap();
            let pointer = machine
                .arena()
                .host_address(range.physical_extents().next().unwrap().start, bytes)
                .unwrap();
            // Stand in for an admitted plain native store: no checked-writer
            // epoch update, and no racing Rust reference to the guest bytes.
            unsafe {
                crate::raw::write_value(pointer.as_ptr(), bytes as usize, 9);
            }
            let observed = crate::Observation::default();
            machine
                .access_captured(
                    std::slice::from_ref(&range),
                    crate::AccessOperation::ReadObserved(observed.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            let before = machine.tracking_statistics().protection_calls;
            assert!(
                !machine
                    .store_exclusive_resolved(&range, reservation, 7)
                    .unwrap()
            );
            assert!(!observed.is_invalid());
            assert_eq!(machine.tracking_statistics().protection_calls, before);
            let mut value = [0; 16];
            machine
                .capture_stream(&[range], &mut value[..bytes as usize])
                .unwrap();
            assert_eq!(u128::from_le_bytes(value), 9);
        }
    }

    #[test]
    fn failed_exclusive_keeps_clean_observers_and_recycling_rejects_old_reservations() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let account = machine.account(PAGE_SIZE).unwrap();
        let allocation = account.allocate(PAGE_SIZE).unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 16).unwrap();
        let (_, old) = machine.load_exclusive_resolved(&range).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&range), &[4; 16])
            .unwrap();
        let observed = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&range),
                crate::AccessOperation::ReadObserved(observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!machine.store_exclusive_resolved(&range, old, 9).unwrap());
        assert!(!observed.is_invalid());
        let (_, current) = machine.load_exclusive_resolved(&range).unwrap();
        assert!(
            machine
                .store_exclusive_resolved(&range, current, 17)
                .unwrap()
        );
        assert!(observed.is_invalid());
        drop(allocation);
        drop(range);
        assert_eq!(machine.tracking_statistics().reservation_metadata_bytes, 0);
        let next = account.allocate(PAGE_SIZE).unwrap();
        let range = AllocationRange::new(next.pin(), 0, 16).unwrap();
        assert!(!machine.store_exclusive_resolved(&range, old, 100).unwrap());
        let (value, current) = machine.load_exclusive_resolved(&range).unwrap();
        assert_eq!(value, 0);
        assert_ne!(current.monitor, old.monitor);
    }
    #[test]
    fn atomic_transforms_use_exact_width_and_compare_failures_preserve_observers() {
        use crate::{
            AccessOperation, AccessOrdering, AccessResult, AtomicAccess, AtomicOperation,
            AtomicTransform,
        };
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 1).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&range), &[0xff])
            .unwrap();
        let observed = crate::Observation::default();
        machine
            .access_captured(
                std::slice::from_ref(&range),
                crate::AccessOperation::ReadObserved(observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let access = AtomicAccess {
            bytes: 1,
            ordering: AccessOrdering::AcquireRelease,
            operation: AtomicOperation::CompareExchange {
                expected: 2,
                replacement: 3,
            },
        };
        assert_eq!(
            machine.atomic_physical(&range, access).unwrap(),
            AccessResult::Atomic {
                previous: 0xff,
                written: false
            }
        );
        assert!(!observed.is_invalid());
        for (transform, operand, expected) in [
            (AtomicTransform::SignedMaximum, 127, 127),
            (AtomicTransform::SignedMinimum, 255, 255),
            (AtomicTransform::Add, 2, 1),
            (AtomicTransform::UnsignedMaximum, 200, 200),
            (AtomicTransform::UnsignedMinimum, 7, 7),
            (AtomicTransform::Xor, 3, 4),
            (AtomicTransform::Set, 2, 6),
            (AtomicTransform::Clear, 4, 2),
            (AtomicTransform::Swap, 0x1234, 0x34),
        ] {
            let access = AtomicAccess {
                operation: AtomicOperation::ReadModifyWrite { transform, operand },
                ..access
            };
            assert!(matches!(
                machine.atomic_physical(&range, access).unwrap(),
                AccessResult::Atomic { written: true, .. }
            ));
            let read = machine
                .apply_resolved_access(std::slice::from_ref(&range), &AccessOperation::Read)
                .unwrap();
            assert_eq!(read, AccessResult::Read(Box::new([expected])));
        }
        assert!(observed.is_invalid());
        for width in [2, 4, 8, 16] {
            let range = AllocationRange::new(allocation.pin(), 0, width).unwrap();
            machine
                .write_resolved(std::slice::from_ref(&range), &vec![0xff; width as usize])
                .unwrap();
            let access = AtomicAccess {
                bytes: width as u8,
                operation: AtomicOperation::ReadModifyWrite {
                    transform: AtomicTransform::Add,
                    operand: 1,
                },
                ..access
            };
            let expected = u128::MAX >> (128 - width * 8);
            assert_eq!(
                machine.atomic_physical(&range, access).unwrap(),
                AccessResult::Atomic {
                    previous: expected,
                    written: true
                }
            );
            let mut bytes = vec![1; width as usize];
            machine.capture_stream(&[range], &mut bytes).unwrap();
            assert_eq!(bytes, vec![0; width as usize]);
        }
    }
    #[test]
    fn concurrent_physical_atomics_have_one_modification_order() {
        use crate::{AccessOrdering, AccessResult, AtomicAccess, AtomicOperation, AtomicTransform};
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 8).unwrap();
        machine
            .prepare_reservations(std::slice::from_ref(&range))
            .unwrap();
        let operation = AtomicAccess {
            bytes: 8,
            ordering: AccessOrdering::Relaxed,
            operation: AtomicOperation::ReadModifyWrite {
                transform: AtomicTransform::Add,
                operand: 1,
            },
        };
        let cores: Vec<_> = (0..4)
            .map(|_| {
                let machine = machine.clone();
                let range = range.clone();
                thread::spawn(move || {
                    (0..2000)
                        .map(|_| {
                            let AccessResult::Atomic {
                                previous,
                                written: true,
                            } = machine.atomic_physical(&range, operation).unwrap()
                            else {
                                panic!("atomic increment failed")
                            };
                            previous
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut previous: Vec<_> = cores
            .into_iter()
            .flat_map(|core| core.join().unwrap())
            .collect();
        previous.sort_unstable();
        assert_eq!(previous, (0..8000).collect::<Vec<_>>());
    }
    #[test]
    fn checked_atomics_and_host_cas_share_one_modification_order() {
        use crate::{AccessOrdering, AccessResult, AtomicAccess, AtomicOperation, AtomicTransform};
        for bytes in [8, 16] {
            let machine = Machine::new(PAGE_SIZE).unwrap();
            let allocation = machine
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap();
            let range = AllocationRange::new(allocation.pin(), 0, bytes).unwrap();
            machine
                .prepare_reservations(std::slice::from_ref(&range))
                .unwrap();
            let workers: Vec<_> = (0..4)
                .map(|index| {
                    let machine = machine.clone();
                    let range = range.clone();
                    thread::spawn(move || {
                        let pointer = machine
                            .arena()
                            .host_address(range.physical_extents().next().unwrap().start, bytes)
                            .unwrap();
                        (0..2000)
                            .map(|_| {
                                if index < 2 {
                                    let operation = AtomicAccess {
                                        bytes: bytes as u8,
                                        ordering: AccessOrdering::Relaxed,
                                        operation: AtomicOperation::ReadModifyWrite {
                                            transform: AtomicTransform::Add,
                                            operand: 1,
                                        },
                                    };
                                    let AccessResult::Atomic {
                                        previous,
                                        written: true,
                                    } = machine.atomic_physical(&range, operation).unwrap()
                                    else {
                                        panic!("increment failed")
                                    };
                                    previous
                                } else {
                                    // Exercise a host atomic that does not participate
                                    // in the Rust reservation/version lock protocol.
                                    let mut previous = unsafe {
                                        crate::raw::read_value(pointer.as_ptr(), bytes as usize)
                                    };
                                    loop {
                                        let observed = unsafe {
                                            crate::raw::compare_exchange(
                                                pointer.as_ptr(),
                                                bytes as usize,
                                                previous,
                                                previous + 1,
                                            )
                                        };
                                        if observed == previous {
                                            break previous;
                                        }
                                        previous = observed;
                                    }
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut previous: Vec<_> = workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect();
            previous.sort_unstable();
            assert_eq!(previous, (0..8000).collect::<Vec<_>>());
        }
    }

    #[test]
    fn scattered_capture_observes_once_and_preserves_uncopied_destination_bytes() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
        let physical = input.physical_extents().next().unwrap().start;
        let first = AllocationRange::new(allocation.pin(), 7, 3).unwrap();
        let second = AllocationRange::new(allocation.pin(), PAGE_SIZE + 5, 4).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&first), &[1, 2, 3])
            .unwrap();
        machine
            .write_resolved(std::slice::from_ref(&second), &[4, 5, 6, 7])
            .unwrap();
        let copies = [
            PhysicalCapture {
                source: Extent {
                    start: ArenaOffset::new(physical.get() + 7),
                    bytes: 3,
                },
                destination: 20,
            },
            PhysicalCapture {
                source: Extent {
                    start: ArenaOffset::new(physical.get() + PAGE_SIZE + 5),
                    bytes: 4,
                },
                destination: 0,
            },
        ];
        let mut bytes = [0xcc; 32];
        let observation = machine
            .capture_clean_scattered(&[input.clone(), input.clone()], &copies, &mut bytes)
            .unwrap();
        assert_eq!(&bytes[0..4], &[4, 5, 6, 7]);
        assert_eq!(&bytes[20..23], &[1, 2, 3]);
        assert!(
            bytes[4..20]
                .iter()
                .chain(&bytes[23..])
                .all(|byte| *byte == 0xcc)
        );
        assert_eq!(machine.tracking_statistics().observer_registrations, 2);
        assert!(!observation.is_invalid());
        let before = machine.tracking_statistics();
        let invalid = PhysicalCapture {
            source: Extent {
                start: ArenaOffset::new(physical.get() + 2 * PAGE_SIZE),
                bytes: 1,
            },
            destination: 0,
        };
        assert!(
            machine
                .capture_clean_scattered(std::slice::from_ref(&input), &[invalid], &mut bytes)
                .is_err()
        );
        assert!(
            machine
                .capture_clean_scattered(std::slice::from_ref(&input), &copies, &mut bytes[..2])
                .is_err()
        );
        assert_eq!(machine.tracking_statistics(), before);
        machine.write_resolved(&[first], &[8, 9, 10]).unwrap();
        assert!(observation.is_invalid());
        let observation = machine
            .capture_clean_scattered(std::slice::from_ref(&input), &copies, &mut bytes)
            .unwrap();
        drop(second);
        drop(input);
        drop(allocation);
        assert!(observation.is_invalid());
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }
}
