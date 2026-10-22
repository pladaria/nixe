//! Machine allocations retain physical extents independently of guest ownership.
use crate::heap::PageHeap;
use crate::{AllocationId, ArenaOffset, Extent, MachineId, PAGE_SIZE, PhysicalArena};
mod ownership;
mod pressure;
pub use ownership::AllocationObservation;
use ownership::{PhysicalPin, Reference, References};
pub use pressure::AllocationFuture;
use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

#[derive(Debug)]
pub enum MemoryError {
    Host(std::io::Error),
    Invalid(&'static str),
    Unmapped(crate::GuestVirtualAddress),
    PermissionDenied {
        address: crate::GuestVirtualAddress,
        required: crate::MemoryPermissions,
    },
    AddressRange {
        start: crate::GuestVirtualAddress,
        bytes: u64,
    },
    AliasMismatch,
    Execution(Box<str>),
    TransitionFailed(Arc<MemoryError>),
    ProcessLimit {
        requested: u64,
        used: u64,
        limit: u64,
    },
    PhysicalExhaustion {
        requested: u64,
        free: u64,
    },
    RetainedAllocations {
        requested: u64,
        free: u64,
        retained: u64,
    },
}
impl MemoryError {
    /// Guest translation/access rejection, distinct from an internal invariant,
    /// allocation failure or failed host/native operation.
    pub fn is_access_fault(&self) -> bool {
        matches!(
            self,
            Self::Unmapped(_) | Self::PermissionDenied { .. } | Self::AddressRange { .. }
        )
    }
}
impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(error) => write!(f, "host memory: {error}"),
            Self::Invalid(detail) => f.write_str(detail),
            Self::Unmapped(address) => write!(f, "unmapped virtual address {address}"),
            Self::PermissionDenied { address, required } => write!(
                f,
                "guest memory permission denied at {address}: required={required:?}"
            ),
            Self::AddressRange { start, bytes } => write!(
                f,
                "virtual range exceeds the address space: start={start} bytes={bytes}"
            ),
            Self::AliasMismatch => f.write_str("alias does not name the original physical mapping"),
            Self::Execution(detail) => write!(f, "native execution: {detail}"),
            Self::TransitionFailed(error) => write!(f, "address-space transition failed: {error}"),
            Self::ProcessLimit {
                requested,
                used,
                limit,
            } => write!(
                f,
                "process RAM limit: requested={requested} used={used} limit={limit}"
            ),
            Self::PhysicalExhaustion { requested, free } => write!(
                f,
                "physical RAM exhausted: requested={requested} free={free}"
            ),
            Self::RetainedAllocations {
                requested,
                free,
                retained,
            } => write!(
                f,
                "physical RAM waits for retained native/mapping uses: requested={requested} free={free} retained={retained}"
            ),
        }
    }
}
impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Host(error) => Some(error),
            Self::TransitionFailed(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}
impl From<std::io::Error> for MemoryError {
    fn from(error: std::io::Error) -> Self {
        Self::Host(error)
    }
}

struct State {
    heap: PageHeap,
    next_allocation: u64,
    allocations: HashMap<AllocationId, Weak<AllocationRecord>>,
    // Only recycling needs clearing: untouched memfd pages are already zero.
    recycled: Vec<u64>,
}
struct MachineInner {
    id: MachineId,
    arena: PhysicalArena,
    state: Mutex<State>,
    guest_bytes: AtomicU64,
    virtual_reserved: AtomicU64,
    page_table_reserved: AtomicU64,
    aliases: Mutex<crate::aliases::Aliases>,
    tracking: crate::tracking::Tracking,
    cpu_writes: crate::cpu_writes::CpuWrites,
    physical_access: crate::physical_access::PhysicalAccess,
    device_endpoint: Mutex<Option<Weak<dyn crate::DeviceAccessEndpoint>>>,
    next_access: AtomicU64,
    pressure: pressure::AllocationPressure,
}
#[derive(Clone)]
pub struct Machine(Arc<MachineInner>);
impl fmt::Debug for Machine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Machine")
            .field("identity", &self.identity())
            .field("physical_capacity", &self.arena().capacity())
            .finish()
    }
}
#[derive(Clone)]
pub struct MemoryAccount {
    machine: Machine,
    ledger: Arc<Ledger>,
}
struct Ledger {
    limit: u64,
    used: AtomicU64,
}
struct AllocationRecord {
    machine: Machine,
    id: AllocationId,
    extents: Box<[AllocationExtent]>,
    bytes: u64,
    ledger: Arc<Ledger>,
    guest_bytes: AtomicU64,
    live_bytes: Arc<AtomicU64>,
    references: Mutex<References>,
    retirement_observers: Mutex<crate::IntervalIndex<Weak<AtomicBool>>>,
    device_write_epoch: crate::DeviceWriteEpoch,
}
struct AllocationExtent {
    end: u64,
    physical: Extent,
}
struct GuestOwnership {
    record: Arc<AllocationRecord>,
    offset: u64,
    bytes: u64,
}
/// Clones name the same guest ownership and charge; they never allocate RAM.
#[derive(Clone)]
pub struct Allocation {
    ownership: Arc<GuestOwnership>,
}
/// Mappings, IPC and native inputs retain this pin, without retaining a guest
/// accounting charge. One allocation pin can cover all its physical extents.
#[derive(Clone)]
pub struct AllocationPin(Arc<PhysicalPin>);

/// Cached metadata does not keep physical RAM alive after guest ownership and
/// actual captured requests have gone. Native device/upload handles retain
/// their own storage instead of retaining these emulated RAM bytes.
#[derive(Clone)]
pub struct WeakAllocation {
    machine: MachineId,
    allocation: AllocationId,
    record: Weak<AllocationRecord>,
    live_bytes: Arc<AtomicU64>,
    device_write_epoch: crate::DeviceWriteEpoch,
}
impl WeakAllocation {
    /// A reclamation snapshot, without retaining RAM. A consuming operation
    /// must acquire its own range pin; zero-reference pages cannot resurrect.
    pub fn live_ranges(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<std::ops::Range<u64>>, MemoryError> {
        let Some(record) = self.record.upgrade() else {
            return Ok(Vec::new());
        };
        record.references.lock().unwrap().live_ranges(offset, bytes)
    }
    pub fn machine(&self) -> MachineId {
        self.machine
    }
    pub fn identity(&self) -> AllocationId {
        self.allocation
    }
    pub fn upgrade(&self) -> Option<AllocationPin> {
        let record = self.record.upgrade()?;
        record.pin(0, record.bytes).ok()
    }
    pub fn is_live(&self) -> bool {
        self.live_bytes.load(Ordering::Acquire) != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryStatistics {
    pub physical_capacity: u64,
    pub virtual_reserved_bytes: u64,
    pub page_table_reserved_bytes: u64,
    pub allocated_bytes: u64,
    pub guest_owned_bytes: u64,
    pub retained_bytes: u64,
    pub live_allocations: usize,
}
impl Machine {
    pub fn new(capacity: u64) -> Result<Self, MemoryError> {
        static NEXT_MACHINE: AtomicU64 = AtomicU64::new(1);
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("cmpxchg16b") {
            return Err(MemoryError::Invalid(
                "canonical x86-64 memory requires CMPXCHG16B",
            ));
        }
        let heap =
            PageHeap::new(capacity).ok_or(MemoryError::Invalid("invalid physical RAM capacity"))?;
        let id = NEXT_MACHINE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("machine identity exhausted"))?;
        Ok(Self(Arc::new(MachineInner {
            id: MachineId::new(id),
            arena: PhysicalArena::new(capacity)?,
            state: Mutex::new(State {
                heap,
                next_allocation: 1,
                allocations: HashMap::new(),
                recycled: vec![0; (capacity / PAGE_SIZE).div_ceil(64) as usize],
            }),
            guest_bytes: AtomicU64::new(0),
            virtual_reserved: AtomicU64::new(0),
            page_table_reserved: AtomicU64::new(0),
            aliases: Mutex::new(crate::aliases::Aliases::default()),
            tracking: crate::tracking::Tracking::new(capacity),
            cpu_writes: crate::cpu_writes::CpuWrites::default(),
            physical_access: crate::physical_access::PhysicalAccess::new(capacity),
            device_endpoint: Mutex::new(None),
            next_access: AtomicU64::new(1),
            pressure: pressure::AllocationPressure::default(),
        })))
    }
    pub fn switch_ram() -> Result<Self, MemoryError> {
        Self::new(crate::DEFAULT_RAM_BYTES)
    }
    pub fn identity(&self) -> MachineId {
        self.0.id
    }
    pub(crate) fn cpu_writes(&self) -> &crate::cpu_writes::CpuWrites {
        &self.0.cpu_writes
    }
    pub fn arena(&self) -> &PhysicalArena {
        &self.0.arena
    }
    pub fn account(&self, limit: u64) -> Result<MemoryAccount, MemoryError> {
        if limit > self.arena().capacity() {
            return Err(MemoryError::Invalid(
                "process RAM limit exceeds physical machine RAM",
            ));
        }
        Ok(MemoryAccount {
            machine: self.clone(),
            ledger: Arc::new(Ledger {
                limit,
                used: AtomicU64::new(0),
            }),
        })
    }
    pub fn statistics(&self) -> MemoryStatistics {
        let state = self.0.state.lock().unwrap();
        let capacity = self.arena().capacity();
        let allocated = capacity - state.heap.free_bytes();
        let guest = self.0.guest_bytes.load(Ordering::Acquire);
        MemoryStatistics {
            physical_capacity: capacity,
            virtual_reserved_bytes: self.0.virtual_reserved.load(Ordering::Acquire),
            page_table_reserved_bytes: self.0.page_table_reserved.load(Ordering::Acquire),
            allocated_bytes: allocated,
            guest_owned_bytes: guest,
            retained_bytes: allocated - guest,
            live_allocations: state.allocations.len(),
        }
    }
    pub(crate) fn reserve_virtual(&self, bytes: u64) {
        self.0.virtual_reserved.fetch_add(bytes, Ordering::Release);
    }
    pub(crate) fn release_virtual(&self, bytes: u64) {
        self.0.virtual_reserved.fetch_sub(bytes, Ordering::Release);
    }
    pub(crate) fn reserve_page_table(&self, bytes: u64) {
        self.0
            .page_table_reserved
            .fetch_add(bytes, Ordering::Release);
    }
    pub(crate) fn release_page_table(&self, bytes: u64) {
        self.0
            .page_table_reserved
            .fetch_sub(bytes, Ordering::Release);
    }
    pub(crate) fn aliases(&self) -> std::sync::MutexGuard<'_, crate::aliases::Aliases> {
        self.0
            .aliases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub(crate) fn tracking(&self) -> &crate::tracking::Tracking {
        &self.0.tracking
    }
    pub(crate) fn physical_access(&self) -> &crate::physical_access::PhysicalAccess {
        &self.0.physical_access
    }
    pub(crate) fn device_endpoint(&self) -> &Mutex<Option<Weak<dyn crate::DeviceAccessEndpoint>>> {
        &self.0.device_endpoint
    }
    pub(crate) fn next_access_request(&self) -> Result<crate::AccessRequestId, MemoryError> {
        self.0
            .next_access
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map(crate::AccessRequestId)
            .map_err(|_| MemoryError::Invalid("physical access request identity exhausted"))
    }
    pub fn pin(&self, id: AllocationId) -> Option<AllocationPin> {
        let record = self
            .0
            .state
            .lock()
            .unwrap()
            .allocations
            .get(&id)?
            .upgrade()?;
        record.pin(0, record.bytes).ok()
    }
}
impl MemoryAccount {
    pub fn machine(&self) -> &Machine {
        &self.machine
    }
    pub fn used_bytes(&self) -> u64 {
        self.ledger.used.load(Ordering::Acquire)
    }
    pub fn limit(&self) -> u64 {
        self.ledger.limit
    }
    pub fn allocate(&self, bytes: u64) -> Result<Allocation, MemoryError> {
        self.reserve(bytes, None)
    }
    pub fn allocate_contiguous(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Allocation, MemoryError> {
        if alignment < PAGE_SIZE || !alignment.is_power_of_two() {
            return Err(MemoryError::Invalid(
                "physical alignment must be a power of two of at least one page",
            ));
        }
        self.reserve(bytes, Some(alignment))
    }
    fn reserve(&self, bytes: u64, alignment: Option<u64>) -> Result<Allocation, MemoryError> {
        if bytes == 0 || !bytes.is_multiple_of(PAGE_SIZE) {
            return Err(MemoryError::Invalid(
                "physical allocation size must be a nonzero page multiple",
            ));
        }
        // A cold contiguous-pressure query may temporarily retain allocation
        // records. Drop those pins only after the allocator guard, including on
        // an early return: the final pin's destructor takes that same lock.
        let mut retention_pins = Vec::new();
        let (record, clear) = {
            let mut state = self.machine.0.state.lock().unwrap();
            let used = self.used_bytes();
            if bytes > self.limit() - used {
                return Err(MemoryError::ProcessLimit {
                    requested: bytes,
                    used,
                    limit: self.limit(),
                });
            }
            let id = AllocationId::new(state.next_allocation);
            let next = state
                .next_allocation
                .checked_add(1)
                .ok_or(MemoryError::Invalid("allocation identity exhausted"))?;
            let extents = if let Some(alignment) = alignment {
                state
                    .heap
                    .allocate_contiguous(bytes, alignment)
                    .map(|e| vec![e])
            } else {
                state.heap.allocate(bytes)
            };
            let Some(extents) = extents else {
                let free = state.heap.free_bytes();
                let retained = self.machine.arena().capacity()
                    - free
                    - self.machine.0.guest_bytes.load(Ordering::Acquire);
                let mut can_retire = retained != 0 && free + retained >= bytes;
                if can_retire && let Some(alignment) = alignment {
                    // Capacity alone cannot establish contiguous availability.
                    // Reuse the allocator's geometry on cold bitmap metadata;
                    // guest bytes are neither retained nor copied by this query.
                    let mut potential = state.heap.clone();
                    let mut retiring = false;
                    for weak in state.allocations.values() {
                        if let Some(record) = weak.upgrade() {
                            {
                                // SAFETY: these exact extents are allocated in
                                // the private bitmap copy, and are freed once.
                                let retained = record
                                    .physical_ranges(&record.references.lock().unwrap().retained());
                                unsafe {
                                    potential.release(&retained);
                                }
                            }
                            retention_pins.push(record);
                        } else {
                            // Final destruction has started and will free its
                            // extents immediately after this guard is released.
                            retiring = true;
                        }
                    }
                    can_retire =
                        potential.allocate_contiguous(bytes, alignment).is_some() || retiring;
                }
                return Err(if can_retire {
                    MemoryError::RetainedAllocations {
                        requested: bytes,
                        free,
                        retained,
                    }
                } else {
                    MemoryError::PhysicalExhaustion {
                        requested: bytes,
                        free,
                    }
                });
            };
            state.next_allocation = next;
            let mut clear: Vec<Extent> = Vec::new();
            for extent in &extents {
                let first = extent.start.get() / PAGE_SIZE;
                let end = first + extent.bytes / PAGE_SIZE;
                for word in first / 64..end.div_ceil(64) {
                    let low = first.saturating_sub(word * 64);
                    let high = (end - word * 64).min(64);
                    let mask = (u64::MAX << low) & (u64::MAX >> (64 - high));
                    let recycled = &mut state.recycled[word as usize];
                    let mut bits = *recycled & mask;
                    *recycled &= !mask;
                    while bits != 0 {
                        let start = bits.trailing_zeros();
                        let count = (bits >> start).trailing_ones();
                        let bytes = u64::from(count) * PAGE_SIZE;
                        let physical = (word * 64 + u64::from(start)) * PAGE_SIZE;
                        if let Some(last) = clear
                            .last_mut()
                            .filter(|e| e.start.get() + e.bytes == physical)
                        {
                            last.bytes += bytes;
                        } else {
                            clear.push(Extent {
                                start: ArenaOffset::new(physical),
                                bytes,
                            });
                        }
                        let mask = (u64::MAX >> (64 - count)) << start;
                        bits &= !mask;
                    }
                }
            }
            self.ledger.used.fetch_add(bytes, Ordering::Release);
            self.machine
                .0
                .guest_bytes
                .fetch_add(bytes, Ordering::Release);
            let mut end = 0;
            let record = Arc::new(AllocationRecord {
                machine: self.machine.clone(),
                id,
                extents: extents
                    .into_iter()
                    .map(|physical| {
                        end += physical.bytes;
                        AllocationExtent { end, physical }
                    })
                    .collect(),
                bytes,
                ledger: Arc::clone(&self.ledger),
                guest_bytes: AtomicU64::new(bytes),
                live_bytes: Arc::new(AtomicU64::new(bytes)),
                references: Mutex::new(References::new(bytes)),
                retirement_observers: Mutex::new(crate::IntervalIndex::default()),
                device_write_epoch: crate::DeviceWriteEpoch::default(),
            });
            (record, clear)
        };
        let allocation = Allocation {
            ownership: Arc::new(GuestOwnership {
                record: Arc::clone(&record),
                offset: 0,
                bytes,
            }),
        };
        // Clearing can touch many bytes, so it runs outside the allocator lock.
        // The new allocation is not discoverable until clearing has finished.
        for extent in clear {
            unsafe {
                self.machine
                    .arena()
                    .clear_unpublished(extent.start, extent.bytes);
            }
        }
        self.machine
            .0
            .state
            .lock()
            .unwrap()
            .allocations
            .insert(record.id, Arc::downgrade(&record));
        Ok(allocation)
    }
}
impl Allocation {
    pub fn pin(&self) -> AllocationPin {
        self.ownership
            .record
            .pin(self.ownership.offset, self.ownership.bytes)
            .expect("live guest ownership")
    }
    pub fn pin_range(&self, offset: u64, bytes: u64) -> Result<AllocationPin, MemoryError> {
        if !self.owns(offset, bytes) {
            return Err(MemoryError::Invalid("pin exceeds guest ownership"));
        }
        self.ownership.record.pin(offset, bytes)
    }
    pub(crate) fn owns(&self, offset: u64, bytes: u64) -> bool {
        offset >= self.ownership.offset
            && bytes != 0
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.ownership.offset + self.ownership.bytes)
    }
    pub(crate) fn physical_extents(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = Extent> + '_, MemoryError> {
        if !self.owns(offset, bytes) {
            return Err(MemoryError::Invalid(
                "allocation access exceeds guest ownership",
            ));
        }
        Ok(self.ownership.record.physical_extents(offset, bytes))
    }
    pub(crate) fn view(&self, offset: u64, bytes: u64) -> Result<Self, MemoryError> {
        if !self.owns(offset, bytes)
            || !offset.is_multiple_of(PAGE_SIZE)
            || !bytes.is_multiple_of(PAGE_SIZE)
        {
            return Err(MemoryError::Invalid("mapping exceeds guest ownership"));
        }
        if offset == self.ownership.offset && bytes == self.ownership.bytes {
            return Ok(self.clone());
        }
        let record = &self.ownership.record;
        record
            .references
            .lock()
            .unwrap()
            .retain(offset, bytes, Reference::Guest)?;
        Ok(Self {
            ownership: Arc::new(GuestOwnership {
                record: Arc::clone(record),
                offset,
                bytes,
            }),
        })
    }
    pub fn machine(&self) -> &Machine {
        &self.ownership.record.machine
    }
    pub fn identity(&self) -> AllocationId {
        self.ownership.record.id
    }
    pub fn bytes(&self) -> u64 {
        self.ownership.record.bytes
    }
}
impl WeakAllocation {
    pub fn device_write_epoch(&self) -> Option<crate::DeviceWriteEpoch> {
        self.is_live().then(|| self.device_write_epoch.clone())
    }
    pub fn upgrade_range(&self, offset: u64, bytes: u64) -> Option<AllocationPin> {
        self.record.upgrade()?.pin(offset, bytes).ok()
    }
}
impl AllocationPin {
    pub(crate) fn covers(&self, offset: u64, bytes: u64) -> bool {
        offset >= self.0.offset
            && bytes != 0
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.0.offset + self.0.bytes)
    }
    pub fn pin_range(&self, offset: u64, bytes: u64) -> Result<Self, MemoryError> {
        if !self.covers(offset, bytes) {
            return Err(MemoryError::Invalid("pin exceeds physical retention"));
        }
        self.0.record.pin(offset, bytes)
    }
    pub fn device_write_epoch(&self) -> crate::DeviceWriteEpoch {
        self.0.record.device_write_epoch.clone()
    }
    pub(crate) fn mark_device_modified(&self) {
        self.0.record.device_write_epoch.mark_device_modified();
    }
    pub fn downgrade(&self) -> WeakAllocation {
        WeakAllocation {
            machine: self.machine().identity(),
            allocation: self.identity(),
            record: Arc::downgrade(&self.0.record),
            live_bytes: Arc::clone(&self.0.record.live_bytes),
            device_write_epoch: self.0.record.device_write_epoch.clone(),
        }
    }
    pub fn identity(&self) -> AllocationId {
        self.0.record.id
    }
    pub fn machine(&self) -> &Machine {
        &self.0.record.machine
    }
    pub fn bytes(&self) -> u64 {
        self.0.record.bytes
    }
    #[inline]
    pub(crate) fn physical_extents(
        &self,
        offset: u64,
        bytes: u64,
    ) -> impl Iterator<Item = Extent> + '_ {
        self.0.record.physical_extents(offset, bytes)
    }
}
impl Drop for GuestOwnership {
    fn drop(&mut self) {
        let record = &self.record;
        // Most alias/view releases cannot change accounting. They do not need
        // the allocator writer while a mapping/protection writer is active.
        {
            let mut references = record.references.lock().unwrap();
            if references.can_release_guest_without_accounting(self.offset, self.bytes) {
                let released = references.release(self.offset, self.bytes, Reference::Guest);
                debug_assert_eq!(released.guest, 0);
                debug_assert!(released.physical.is_empty());
                return;
            }
        }
        // Lock order for accounting is allocator -> reference metadata. Pin
        // release drops reference metadata before recycling any physical run.
        let state = record.machine.0.state.lock().unwrap();
        let released =
            record
                .references
                .lock()
                .unwrap()
                .release(self.offset, self.bytes, Reference::Guest);
        let guest_before = record
            .guest_bytes
            .fetch_sub(released.guest, Ordering::AcqRel);
        record
            .ledger
            .used
            .fetch_sub(released.guest, Ordering::AcqRel);
        record
            .machine
            .0
            .guest_bytes
            .fetch_sub(released.guest, Ordering::AcqRel);
        let last_guest = released.guest != 0 && guest_before == released.guest;
        drop(state);
        if last_guest {
            let endpoint = record
                .machine
                .device_endpoint()
                .lock()
                .unwrap()
                .as_ref()
                .and_then(Weak::upgrade);
            if let Some(endpoint) = endpoint {
                endpoint.allocation_released(record.machine.identity(), record.id);
            }
        }
        record.retire(&released.physical);
    }
}
impl Drop for AllocationRecord {
    fn drop(&mut self) {
        debug_assert_eq!(self.live_bytes.load(Ordering::Acquire), 0);
        debug_assert_eq!(self.guest_bytes.load(Ordering::Acquire), 0);
        self.machine
            .0
            .state
            .lock()
            .unwrap()
            .allocations
            .remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_guest_owner_notifies_caches_without_locks_or_retaining_released_ram() {
        struct Endpoint {
            machine: Machine,
            released: Mutex<Vec<AllocationId>>,
            retired: Mutex<Vec<AllocationId>>,
        }
        impl crate::DeviceAccessEndpoint for Endpoint {
            fn submit(&self, _: crate::DeviceAccessRequest) {
                panic!("this test has no device accesses");
            }
            fn cancel(&self, _: crate::AccessRequestId) {
                panic!("this test has no cancelled accesses");
            }
            fn allocation_pressure(&self, _: crate::MachineId, _: u64) {}
            fn allocation_released(&self, machine: MachineId, allocation: AllocationId) {
                assert_eq!(machine, self.machine.identity());
                assert!(self.machine.0.state.try_lock().is_ok());
                assert!(self.machine.device_endpoint().try_lock().is_ok());
                let pin = self.machine.pin(allocation).unwrap();
                assert_eq!(pin.0.record.guest_bytes.load(Ordering::Acquire), 0);
                assert!(pin.0.record.references.try_lock().is_ok());
                self.released.lock().unwrap().push(allocation);
            }
            fn physical_retired(
                &self,
                machine: MachineId,
                allocation: AllocationId,
                extents: Box<[Extent]>,
            ) {
                assert_eq!(machine, self.machine.identity());
                assert!(self.machine.0.state.try_lock().is_ok());
                assert!(self.machine.device_endpoint().try_lock().is_ok());
                assert!(self.machine.pin(allocation).is_none());
                assert_eq!(extents.iter().map(|e| e.bytes).sum::<u64>(), PAGE_SIZE);
                self.retired.lock().unwrap().push(allocation);
            }
        }
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let endpoint = Arc::new(Endpoint {
            machine: machine.clone(),
            released: Mutex::new(Vec::new()),
            retired: Mutex::new(Vec::new()),
        });
        let route: Arc<dyn crate::DeviceAccessEndpoint> = endpoint.clone();
        machine.set_device_endpoint(&route);
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let id = allocation.identity();
        let alias = allocation.clone();
        let pin = allocation.pin();
        let weak = pin.downgrade();
        drop(allocation);
        assert!(endpoint.released.lock().unwrap().is_empty());
        drop(alias);
        assert_eq!(*endpoint.released.lock().unwrap(), [id]);
        assert_eq!(machine.statistics().guest_owned_bytes, 0);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        assert!(endpoint.retired.lock().unwrap().is_empty());
        drop(pin);
        assert!(!weak.is_live());
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(*endpoint.released.lock().unwrap(), [id]);
        assert_eq!(*endpoint.retired.lock().unwrap(), [id]);
    }
    #[test]
    fn accounts_share_one_machine_and_alias_pins_do_not_charge_again() {
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let a = machine.account(8 * PAGE_SIZE).unwrap();
        let b = machine.account(8 * PAGE_SIZE).unwrap();
        let owned = a.allocate(8 * PAGE_SIZE).unwrap();
        let alias = owned.clone();
        let pin = owned.pin();
        let other = b.allocate(8 * PAGE_SIZE).unwrap();
        assert_eq!(machine.statistics().allocated_bytes, 16 * PAGE_SIZE);
        assert_eq!(a.used_bytes(), 8 * PAGE_SIZE);
        drop(owned);
        drop(alias);
        assert_eq!(a.used_bytes(), 0);
        assert_eq!(machine.statistics().retained_bytes, 8 * PAGE_SIZE);
        assert!(matches!(
            a.allocate(PAGE_SIZE),
            Err(MemoryError::RetainedAllocations { .. })
        ));
        drop(pin);
        assert_eq!(machine.statistics().allocated_bytes, 8 * PAGE_SIZE);
        drop(other);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(machine.statistics().live_allocations, 0);
    }
    #[test]
    fn recycled_bytes_are_cleared_after_the_last_pin_and_identity_never_repeats() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(2 * PAGE_SIZE).unwrap();
        let id = first.identity();
        let pin = first.pin();
        let address = pin
            .machine()
            .arena()
            .host_address(
                pin.physical_extents(0, pin.bytes()).next().unwrap().start,
                2 * PAGE_SIZE,
            )
            .unwrap();
        unsafe {
            address.as_ptr().add(19).write(117);
        }
        drop(first);
        assert!(machine.pin(id).is_some());
        drop(pin);
        assert!(machine.pin(id).is_none());
        let second = account.allocate(2 * PAGE_SIZE).unwrap();
        assert_ne!(second.identity(), id);
        assert_eq!(unsafe { address.as_ptr().add(19).read() }, 0);
        drop(second);
    }
    #[test]
    fn account_limits_and_physical_limits_remain_distinct() {
        let machine = Machine::new(8 * PAGE_SIZE).unwrap();
        assert!(machine.account(9 * PAGE_SIZE).is_err());
        let account = machine.account(PAGE_SIZE).unwrap();
        assert!(matches!(
            account.allocate(2 * PAGE_SIZE),
            Err(MemoryError::ProcessLimit { .. })
        ));
        let full = machine
            .account(8 * PAGE_SIZE)
            .unwrap()
            .allocate(8 * PAGE_SIZE)
            .unwrap();
        assert!(matches!(
            account.allocate(PAGE_SIZE),
            Err(MemoryError::PhysicalExhaustion { .. })
        ));
        drop(full);
    }
}
