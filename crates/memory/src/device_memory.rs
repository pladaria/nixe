//! Shared device virtual memory, independent of GPU VA and NvMap heap policy.
//! Backing metadata is weak; live device mappings own the accepted RAM pins.
//! CPU physical continuity remains distinct from mapped device coverage.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
use crate::PhysicalMapping;
use crate::{AllocationId, AllocationRange, Machine, MemoryError, PAGE_SIZE};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Mutex, OnceLock, RwLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering},
    },
};
mod reader;
mod reverse;
use reverse::{AliasChains, INDEX_MASK, MULTIPLE};
mod table;
mod writer;
pub use reader::DeviceMemoryReader;
use table::TrackedTable;
const LEAF_SHIFT: u32 = 21;
const LEAF_PAGES: usize = 1 << (LEAF_SHIFT - 12);

// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/range_mutex.h
#[derive(Debug, Default)]
struct CachedRangeMutex {
    ranges: Mutex<Vec<(u64, u64)>>,
    changed: Condvar,
}
struct CachedRangeGuard<'a> {
    owner: &'a CachedRangeMutex,
    start: u64,
    end: u64,
}
impl CachedRangeMutex {
    fn lock(&self, start: u64, end: u64) -> CachedRangeGuard<'_> {
        let mut ranges = self.ranges.lock().unwrap();
        while ranges.iter().any(|&(a, b)| start < b && a < end) {
            ranges = self.changed.wait(ranges).unwrap();
        }
        ranges.push((start, end));
        CachedRangeGuard {
            owner: self,
            start,
            end,
        }
    }
}
impl Drop for CachedRangeGuard<'_> {
    fn drop(&mut self) {
        let mut ranges = self.owner.ranges.lock().unwrap();
        let index = ranges
            .iter()
            .position(|&(a, b)| a == self.start && b == self.end)
            .expect("held cached-page range");
        ranges.swap_remove(index);
        drop(ranges);
        self.owner.changed.notify_all();
    }
}

/// The selected rasterizer's actual cache owners receive device invalidations
/// before the old CPU backing disappears. This is not an owner-thread queue.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L244-L270
pub trait DeviceMemoryInterface: Send + Sync + std::fmt::Debug {
    fn must_flush_region(&self, address: u64, bytes: u64) -> bool;
    /// Resolve an already available read without blocking native execution.
    /// Return only preemptive areas. A reactive read must remain pending and
    /// must not be marked flushed until on_cpu_read requests its publication.
    fn try_on_cpu_read(&self, address: u64, bytes: u64) -> Option<crate::NativeReadArea>;
    fn on_cpu_read(&self, address: u64, bytes: u64) -> crate::NativeReadArea;
    fn invalidate_region(&self, address: u64, bytes: u64);
    fn unmap_memory(&self, address: u64, bytes: u64);
    fn on_cpu_write(&self, address: u64, bytes: u64) -> bool;
}
#[derive(Clone, Copy, Debug, Default)]
struct Page {
    allocation: AllocationId,
    allocation_page: u32,
    users: u32,
}

#[derive(Debug)]
struct State {
    pages: Box<[Option<Box<[Page; LEAF_PAGES]>>]>,
    reservations: BTreeMap<u64, u64>,
    sources: BTreeMap<u64, Weak<PhysicalMapping>>,
    aliases: AliasChains,
    changed: crate::RangeSet,
    processes: Vec<Weak<crate::AddressSpace>>,
}
impl State {
    fn register_process(&mut self, process: &Arc<crate::AddressSpace>) -> u32 {
        let weak = Arc::downgrade(process);
        if let Some(index) = self.processes.iter().position(|entry| entry.ptr_eq(&weak)) {
            return index as u32;
        }
        let index = u32::try_from(self.processes.len()).expect("SMMU process table capacity");
        self.processes.push(weak);
        index
    }
    fn add_reverse(&mut self, reverse: &[AtomicU32], physical: usize, address: u64) {
        // Page+1 preserves the existing zero-address mapping contract. The
        // upper bit selects the indexed multi-address container, as in Eden.
        let page = u32::try_from((address >> 12) + 1).expect("device page index");
        debug_assert_eq!(page & MULTIPLE, 0);
        let head = reverse[physical].load(Ordering::Relaxed);
        if head == 0 {
            reverse[physical].store(page, Ordering::Release);
        } else if head & MULTIPLE != 0 {
            self.aliases.append(head & INDEX_MASK, page);
        } else {
            let first = self.aliases.register(head);
            self.aliases.append(first, page);
            reverse[physical].store(MULTIPLE | first, Ordering::Release);
        }
    }
    fn remove_reverse(&mut self, reverse: &[AtomicU32], physical: usize, address: u64) {
        let page = ((address >> 12) + 1) as u32;
        let head = reverse[physical].load(Ordering::Relaxed);
        let next = if head & MULTIPLE != 0 {
            self.aliases.unregister(head & INDEX_MASK, page)
        } else {
            debug_assert_eq!(head, page);
            0
        };
        reverse[physical].store(next, Ordering::Release);
    }
    fn page(&self, address: u64) -> Page {
        self.pages
            .get((address >> LEAF_SHIFT) as usize)
            .and_then(Option::as_ref)
            .map_or(Page::default(), |leaf| {
                leaf[((address >> 12) as usize) & (LEAF_PAGES - 1)]
            })
    }
    fn page_mut(&mut self, address: u64) -> &mut Page {
        &mut self.pages[(address >> LEAF_SHIFT) as usize]
            .get_or_insert_with(|| Box::new([Page::default(); LEAF_PAGES]))
            [((address >> 12) as usize) & (LEAF_PAGES - 1)]
    }
}
#[derive(Debug)]
pub(crate) struct Inner {
    machine: Machine,
    address_bits: u32,
    state: RwLock<State>,
    tracked: TrackedTable,
    // The compressed physical-to-device index is the canonical reverse table.
    // ApplyOpOnPAddr must not retain the mapping guard during rasterizer callbacks.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.h#L75-L89
    reverse: Box<[AtomicU32]>,
    // Monotonic physical producer history. False proves no native cache/query
    // has ever owned these bytes; true still requires consulting actual caches.
    gpu_written: Box<[AtomicU64]>,
    command_versions: Box<[OnceLock<Box<CommandVersions>>]>,
    command_blocks: Box<[AtomicU64]>,
    has_command_pages: AtomicBool,
    // Pin coverage is serialized independently of table readers. Cache
    // invalidation may inspect the original backing before taking its writer.
    pin_guard: Mutex<()>,
    interface: RwLock<Option<Weak<dyn DeviceMemoryInterface>>>,
    generation: AtomicU64,
    counts: Box<[OnceLock<Box<[AtomicU8; LEAF_PAGES]>>]>,
    counter_guard: CachedRangeMutex,
}
#[derive(Debug)]
struct CommandVersions {
    mask: AtomicU64,
    // Low bit: the full physical page is synchronized to RAM. Higher bits:
    // content generation. Host stores preserve synchronization; GPU producers
    // and mapping retirement invalidate it atomically with the generation.
    versions: [AtomicU64; 64],
}
#[derive(Clone, Debug)]
pub struct DeviceMemory(pub(crate) Arc<Inner>);
#[derive(Debug)]
pub struct DeviceReservation {
    memory: DeviceMemory,
    address: u64,
}
impl DeviceReservation {
    pub fn address(&self) -> u64 {
        self.address
    }
}
impl Drop for DeviceReservation {
    fn drop(&mut self) {
        let mut state = self.memory.0.state.write().unwrap();
        state.reservations.remove(&self.address);
        state.sources.remove(&self.address);
    }
}

#[derive(Debug)]
pub struct DeviceMapping {
    memory: DeviceMemory,
    address: u64,
    bytes: u64,
    physical: Arc<PhysicalMapping>,
    ranges: Arc<[AllocationRange]>,
    tracked: bool,
}
impl DeviceMapping {
    pub fn address(&self) -> u64 {
        self.address
    }
    pub fn physical(&self) -> &Arc<PhysicalMapping> {
        &self.physical
    }
    pub fn ranges(&self) -> &[AllocationRange] {
        &self.ranges
    }
}
impl Drop for DeviceMapping {
    fn drop(&mut self) {
        let _pins = self.memory.0.pin_guard.lock().unwrap();
        let interface = self
            .memory
            .0
            .interface
            .read()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade);
        if let Some(interface) = interface {
            let mut retiring = crate::RangeSet::default();
            {
                let state = self.memory.0.state.read().unwrap();
                for at in (self.address..self.address + self.bytes).step_by(PAGE_SIZE as usize) {
                    if state.page(at).users == 1 {
                        retiring.insert(at, PAGE_SIZE);
                    }
                }
            }
            for (address, bytes) in retiring.iter() {
                interface.unmap_memory(address, bytes);
            }
        }
        let mut state = self.memory.0.state.write().unwrap();
        let mut changed = false;
        for at in (self.address..self.address + self.bytes).step_by(PAGE_SIZE as usize) {
            let page = state.page_mut(at);
            assert!(page.users != 0);
            page.users -= 1;
            let retired = page.users == 0;
            let physical = self.memory.0.tracked.entry(at).page().physical - 1;
            let physical = physical as usize;
            if retired {
                changed = true;
                self.memory.0.tracked.entry(at).set_cpu_backing(0, 0);
                self.memory.0.tracked.entry(at).set_physical(0);
                state.remove_reverse(&self.memory.0.reverse, physical, at);
                state.changed.insert(at, PAGE_SIZE);
            }
        }
        if self.tracked {
            state.sources.remove(&self.address);
        }
        if changed {
            self.memory.0.generation.fetch_add(1, Ordering::Release);
        }
    }
}
impl DeviceMemory {
    pub fn new(machine: Machine, address_bits: u32) -> Self {
        assert!((LEAF_SHIFT..64).contains(&address_bits));
        let physical_pages = (machine.arena().capacity() / PAGE_SIZE) as usize;
        let tracked = TrackedTable::new(&machine, address_bits);
        Self(Arc::new(Inner {
            machine,
            tracked,
            address_bits,
            state: RwLock::new(State {
                pages: (0..1usize << (address_bits - LEAF_SHIFT))
                    .map(|_| None)
                    .collect(),
                reservations: BTreeMap::new(),
                sources: BTreeMap::new(),
                aliases: AliasChains::default(),
                changed: Default::default(),
                processes: vec![Weak::new()],
            }),
            reverse: (0..physical_pages).map(|_| AtomicU32::new(0)).collect(),
            gpu_written: (0..physical_pages.div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            command_versions: (0..physical_pages.div_ceil(64))
                .map(|_| OnceLock::new())
                .collect(),
            command_blocks: (0..physical_pages.div_ceil(4096))
                .map(|_| AtomicU64::new(0))
                .collect(),
            has_command_pages: AtomicBool::new(false),
            generation: AtomicU64::new(1),
            counts: (0..1usize << (address_bits - LEAF_SHIFT))
                .map(|_| OnceLock::new())
                .collect(),
            counter_guard: CachedRangeMutex::default(),
            pin_guard: Mutex::new(()),
            interface: RwLock::new(None),
        }))
    }
    pub fn reader(&self) -> DeviceMemoryReader {
        DeviceMemoryReader::new(self.clone())
    }
    /// Record producers by physical page so aliases and address reuse cannot
    /// turn a written page into a false clean result. History is never cleared.
    #[inline]
    pub fn mark_gpu_written(&self, address: u64, bytes: u64) {
        self.visit_physical_pages(address, bytes, |page| {
            let word = &self.0.gpu_written[page / 64];
            let mask = 1 << (page % 64);
            if word.load(Ordering::Relaxed) & mask == 0 {
                word.fetch_or(mask, Ordering::Release);
            }
            if let Some(versions) = self.0.command_versions[page / 64].get() {
                let version = &versions.versions[page % 64];
                if version.load(Ordering::Relaxed) != 0 {
                    let _ = version.fetch_update(Ordering::Release, Ordering::Relaxed, |state| {
                        Some((state.wrapping_add(2)) & !1)
                    });
                }
            }
            false
        });
    }
    /// Physical identity and producer versions survive SMMU aliases. Allocate
    /// counters only for pages actually consumed as command lists.
    pub fn register_command_pages(&self, address: u64, bytes: u64) -> bool {
        let mut written = false;
        self.visit_physical_pages(address, bytes, |page| {
            if self.0.gpu_written[page / 64].load(Ordering::Acquire) & (1 << (page % 64)) == 0 {
                return false;
            }
            written = true;
            let versions = self.0.command_versions[page / 64].get_or_init(|| {
                Box::new(CommandVersions {
                    mask: AtomicU64::new(0),
                    versions: std::array::from_fn(|_| AtomicU64::new(0)),
                })
            });
            if versions.versions[page % 64].load(Ordering::Acquire) == 0 {
                let _ = versions.versions[page % 64].compare_exchange(
                    0,
                    2,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                );
                versions.mask.fetch_or(1 << (page % 64), Ordering::Release);
                self.0.command_blocks[page / 4096]
                    .fetch_or(1 << (page / 64 % 64), Ordering::Release);
                self.0.has_command_pages.store(true, Ordering::Release);
            }
            false
        });
        written
    }
    pub fn command_version(&self, physical_page: u64) -> u64 {
        self.command_page_state(physical_page).0
    }
    pub fn command_page_state(&self, physical_page: u64) -> (u64, bool) {
        let page = (physical_page / PAGE_SIZE) as usize;
        let state = self
            .0
            .command_versions
            .get(page / 64)
            .and_then(OnceLock::get)
            .map_or(0, |versions| {
                versions.versions[page % 64].load(Ordering::Acquire)
            });
        (state >> 1, state & 1 != 0)
    }
    /// Walk only registered physical command pages. The upper bitmap skips
    /// empty 16 MiB ranges; the leaves select individual pages. Both indexes
    /// are monotonic, so aliases and mapping retirement cannot clear a writer.
    pub fn visit_command_pages(&self, address: u64, bytes: u64, mut visit: impl FnMut(u64, u64)) {
        if bytes == 0 || !self.0.has_command_pages.load(Ordering::Acquire) {
            return;
        }
        let end = address + bytes;
        let mut at = address & !(PAGE_SIZE - 1);
        while at < end {
            let page = self.0.tracked.entry(at).page();
            if page.physical == 0 {
                at += PAGE_SIZE;
                continue;
            }
            let physical = (page.physical - 1) as usize;
            let available = ((end - at).div_ceil(PAGE_SIZE) as usize)
                .min(page.continuity as usize + 1)
                .min(4096 - physical % 4096);
            let blocks = self.0.command_blocks[physical / 4096].load(Ordering::Acquire)
                >> (physical / 64 % 64);
            if blocks == 0 {
                at += available as u64 * PAGE_SIZE;
                continue;
            }
            let next = (physical / 64 + blocks.trailing_zeros() as usize) * 64;
            if next > physical {
                at += available.min(next - physical) as u64 * PAGE_SIZE;
                continue;
            }
            let count = available.min(64 - physical % 64);
            if let Some(versions) = self.0.command_versions[physical / 64].get() {
                let mask = u64::MAX >> (64 - count);
                let mut bits = (versions.mask.load(Ordering::Acquire) >> (physical % 64)) & mask;
                while bits != 0 {
                    let offset = bits.trailing_zeros() as u64;
                    visit(
                        at + offset * PAGE_SIZE,
                        (physical as u64 + offset) * PAGE_SIZE,
                    );
                    bits &= bits - 1;
                }
            }
            at += count as u64 * PAGE_SIZE;
        }
    }
    pub fn invalidate_command_pages(&self, address: u64, bytes: u64) {
        self.visit_command_pages(address, bytes, |_, physical| {
            let page = (physical / PAGE_SIZE) as usize;
            self.0.command_versions[page / 64].get().unwrap().versions[page % 64]
                .fetch_update(Ordering::Release, Ordering::Relaxed, |state| {
                    Some((state.wrapping_add(2)) & !1)
                })
                .ok();
        });
    }
    /// A RAM store invalidates retained GPU copies without making an already
    /// synchronized page dirty again. Unknown/dirty pages remain unsynchronized.
    /// https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L3428-L3455
    pub fn write_command_pages(&self, address: u64, bytes: u64) {
        self.visit_command_pages(address, bytes, |_, physical| {
            let page = (physical / PAGE_SIZE) as usize;
            self.0.command_versions[page / 64].get().unwrap().versions[page % 64]
                .fetch_add(2, Ordering::Release);
        });
    }
    /// Call only after publishing every byte of the covered physical pages.
    pub fn sync_command_pages(&self, address: u64, bytes: u64) {
        self.visit_command_pages(address, bytes, |_, physical| {
            let page = (physical / PAGE_SIZE) as usize;
            self.0.command_versions[page / 64].get().unwrap().versions[page % 64]
                .fetch_or(1, Ordering::Release);
        });
    }
    pub fn may_be_gpu_written(&self, address: u64, bytes: u64) -> bool {
        self.visit_physical_pages(address, bytes, |page| {
            self.0.gpu_written[page / 64].load(Ordering::Acquire) & (1 << (page % 64)) != 0
        })
    }
    fn visit_physical_pages(
        &self,
        address: u64,
        bytes: u64,
        mut visit: impl FnMut(usize) -> bool,
    ) -> bool {
        if bytes == 0 {
            return false;
        }
        let Some(end) = address
            .checked_add(bytes)
            .filter(|&end| end <= 1 << self.0.address_bits)
        else {
            return true;
        };
        for at in (address & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
            let page = self.0.tracked.entry(at).page().physical;
            if page != 0 && visit((page - 1) as usize) {
                return true;
            }
        }
        false
    }
    pub fn physical_address(&self, address: u64) -> Option<crate::ArenaOffset> {
        if address >= 1 << self.0.address_bits {
            return None;
        }
        let physical = self.0.tracked.entry(address).page().physical;
        (physical != 0).then(|| {
            crate::ArenaOffset::new(
                u64::from(physical - 1) * PAGE_SIZE + (address & (PAGE_SIZE - 1)),
            )
        })
    }
    /// DeviceMemoryManager::GetPointer resolves the compressed physical page
    /// directly. This numeric pointer grants no physical lease or coherence.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L333-L350
    #[inline]
    pub fn pointer(&self, address: u64, bytes: u64) -> Option<crate::HostAddress> {
        self.machine()
            .arena()
            .host_address(self.physical_address(address)?, bytes)
    }
    /// GetSpan checks the device PTE's physical continuity before borrowing RAM.
    /// The machine owns the arena; this does not acquire an allocation pin.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L297-L316
    pub fn span(&self, address: u64, bytes: u64) -> Option<crate::HostAddress> {
        if bytes == 0 || address.checked_add(bytes)? > 1 << self.0.address_bits {
            return None;
        }
        let page = self.0.tracked.entry(address).page();
        let within = address & (PAGE_SIZE - 1);
        if page.physical == 0 || bytes > (u64::from(page.continuity) + 1) * PAGE_SIZE - within {
            return None;
        }
        self.machine().arena().host_address(
            crate::ArenaOffset::new(u64::from(page.physical - 1) * PAGE_SIZE + within),
            bytes,
        )
    }
    pub fn bind_interface(
        &self,
        interface: Weak<dyn DeviceMemoryInterface>,
    ) -> Result<(), MemoryError> {
        let mut selected = self.0.interface.write().unwrap();
        if selected.as_ref().and_then(Weak::upgrade).is_some() {
            return Err(MemoryError::Invalid("SMMU already has a live rasterizer"));
        }
        self.machine().bind_device_memory(self)?;
        *selected = Some(interface);
        Ok(())
    }
    pub(crate) fn interface(&self) -> Option<Arc<dyn DeviceMemoryInterface>> {
        self.0
            .interface
            .read()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade)
    }
    // InnerGatherDeviceAddresses rechecks a multi-address header under the
    // mapping guard. The caller's scratch owns the snapshot during callbacks.
    #[cold]
    fn gather_device_pages(&self, physical: usize, scratch: &mut Vec<u32>) {
        let state = self.0.state.read().unwrap();
        let head = self.0.reverse[physical].load(Ordering::Acquire);
        if head & MULTIPLE != 0 {
            state.aliases.gather(head & INDEX_MASK, scratch);
        } else {
            scratch.clear();
            if head != 0 {
                scratch.push(head);
            }
        }
    }
    /// ApplyOpOnPAddr directly consumes the compressed single destination.
    /// Only genuine aliases borrow scratch and enter InnerGatherDeviceAddresses.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/device_memory_manager.h#L78-L97
    #[inline]
    pub(crate) fn apply_on_physical(
        &self,
        extent: crate::Extent,
        scratch: &RefCell<Vec<u32>>,
        mut operation: impl FnMut(u64, u64),
    ) {
        let end = extent.start.get() + extent.bytes;
        let mut at = extent.start.get();
        while at < end {
            let within = at & (PAGE_SIZE - 1);
            let size = (PAGE_SIZE - within).min(end - at);
            let physical = (at / PAGE_SIZE) as usize;
            let head = self.0.reverse[physical].load(Ordering::Acquire);
            if head & MULTIPLE == 0 {
                if head != 0 {
                    operation(u64::from(head - 1) * PAGE_SIZE + within, size);
                }
            } else {
                let mut scratch = scratch.borrow_mut();
                self.gather_device_pages(physical, &mut scratch);
                for &page in scratch.iter() {
                    operation(u64::from(page - 1) * PAGE_SIZE + within, size);
                }
            }
            at += size;
        }
    }
    pub fn address_bits(&self) -> u32 {
        self.0.address_bits
    }
    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn machine(&self) -> &Machine {
        &self.0.machine
    }
    pub fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }
    pub fn reserve(&self, bytes: u64, alignment: u64) -> Result<DeviceReservation, MemoryError> {
        if bytes == 0 || !alignment.is_power_of_two() || alignment < PAGE_SIZE {
            return Err(MemoryError::Invalid("invalid device address reservation"));
        }
        let mut state = self.0.state.write().unwrap();
        let mut start = alignment;
        for (&base, &length) in &state.reservations {
            if start.checked_add(bytes).is_some_and(|end| end <= base) {
                break;
            }
            start = start.max((base + length).div_ceil(alignment) * alignment);
        }
        if start
            .checked_add(bytes)
            .is_none_or(|end| end > 1 << self.0.address_bits)
        {
            return Err(MemoryError::Invalid("device address space exhausted"));
        }
        state.reservations.insert(start, bytes);
        Ok(DeviceReservation {
            memory: self.clone(),
            address: start,
        })
    }
    /// Track the existing CPU backing's physical runs, including currently
    /// unmapped pages, as DeviceMemoryManager::TrackContinuityImpl does.
    pub fn track_backing(
        &self,
        address: u64,
        physical: &Arc<PhysicalMapping>,
    ) -> Result<(), MemoryError> {
        if physical.machine() != self.machine().identity()
            || !(address | physical.bytes()).is_multiple_of(PAGE_SIZE)
            || address
                .checked_add(physical.bytes())
                .is_none_or(|end| end > 1 << self.0.address_bits)
        {
            return Err(MemoryError::Invalid("invalid device continuity backing"));
        }
        let mut state = self.0.state.write().unwrap();
        self.0.tracked.track_continuity(address, physical);
        state.sources.insert(address, Arc::downgrade(physical));
        Ok(())
    }
    pub fn map(
        &self,
        address: u64,
        ranges: Arc<[AllocationRange]>,
        tracked: bool,
    ) -> Result<DeviceMapping, MemoryError> {
        let physical = PhysicalMapping::new(&ranges)?;
        let bytes = physical.bytes();
        if physical.machine() != self.machine().identity()
            || !(address | bytes).is_multiple_of(PAGE_SIZE)
            || address.checked_add(bytes).is_none_or(|end| {
                end > 1 << self.0.address_bits || end / PAGE_SIZE >= u64::from(MULTIPLE)
            })
            || ranges
                .iter()
                .any(|r| !(r.offset() | r.bytes()).is_multiple_of(PAGE_SIZE))
        {
            return Err(MemoryError::Invalid(
                "SMMU input is foreign, unaligned or out of range",
            ));
        }
        let _pins = self.0.pin_guard.lock().unwrap();
        let mut state = self.0.state.write().unwrap();
        let result = (|| {
            let mut at = address;
            for range in ranges.iter() {
                let mut offset = range.offset();
                for extent in range.physical_extents() {
                    for p in (0..extent.bytes).step_by(PAGE_SIZE as usize) {
                        let old = state.page(at);
                        if old.users == u32::MAX
                            || (old.users != 0
                                && (old.allocation != range.allocation().identity()
                                    || u64::from(old.allocation_page) * PAGE_SIZE != offset
                                    || u64::from(self.0.tracked.entry(at).page().physical - 1)
                                        * PAGE_SIZE
                                        != extent.start.get() + p))
                        {
                            return Err(MemoryError::Invalid(
                                "overlapping SMMU heap mappings disagree on physical RAM",
                            ));
                        }
                        at += PAGE_SIZE;
                        offset += PAGE_SIZE;
                    }
                }
            }
            Ok(())
        })();
        result?;
        if tracked {
            self.0.tracked.track_continuity(address, &physical);
            state.sources.insert(address, Arc::downgrade(&physical));
        }
        let mut at = address;
        let mut changed = false;
        for range in ranges.iter() {
            let (process, cpu) = range.cpu_backing().map_or((0, 0), |(space, address)| {
                (state.register_process(&space), address.get())
            });
            let mut offset = range.offset();
            for extent in range.physical_extents() {
                for p in (0..extent.bytes).step_by(PAGE_SIZE as usize) {
                    let page = state.page_mut(at);
                    let added = page.users == 0;
                    changed |= added;
                    *page = Page {
                        allocation: range.allocation().identity(),
                        allocation_page: (offset / PAGE_SIZE) as u32,
                        users: page.users + 1,
                    };
                    let entry = self.0.tracked.entry(at);
                    entry.set_physical(((extent.start.get() + p) / PAGE_SIZE) as u32 + 1);
                    if added {
                        entry.set_cpu_backing(
                            if cpu == 0 { 0 } else { process },
                            if cpu == 0 {
                                0
                            } else {
                                cpu + offset - range.offset()
                            },
                        );
                    }
                    if added {
                        state.add_reverse(
                            &self.0.reverse,
                            ((extent.start.get() + p) / PAGE_SIZE) as usize,
                            at,
                        );
                        state.changed.insert(at, PAGE_SIZE);
                    }
                    at += PAGE_SIZE;
                    offset += PAGE_SIZE;
                }
            }
        }
        // HeapMapper changes the device table only for newly covered pages.
        if changed {
            self.0.generation.fetch_add(1, Ordering::Release);
        }
        drop(state);
        Ok(DeviceMapping {
            memory: self.clone(),
            address,
            bytes,
            physical,
            ranges,
            tracked,
        })
    }
    /// Canonical DAddr cached-page counts, shared by all buffer/image users.
    /// Only zero/one edges change CPU page types, coalesced by CPU continuity.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L539-L620
    pub fn update_pages_cached_count(
        &self,
        address: u64,
        bytes: u64,
        delta: i8,
    ) -> Result<(), MemoryError> {
        assert!(delta == 1 || delta == -1);
        if bytes == 0 {
            return Ok(());
        }
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= 1 << self.0.address_bits)
            .ok_or(MemoryError::Invalid("cached-page range exceeds SMMU"))?;
        let first = address & !(PAGE_SIZE - 1);
        let end = end.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let _counter = self.0.counter_guard.lock(first, end);
        self.update_pages_cached_count_no_lock(first, end, delta)
    }
    /// Apply the union of changed page ranges under one range lock, as Eden's
    /// DeviceMemoryManager::UpdatePagesCachedBatch. Each page changes once.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/device_memory_manager.inc#L624-L658
    pub fn update_pages_cached_batch(
        &self,
        ranges: &[(u64, u64)],
        delta: i8,
    ) -> Result<(), MemoryError> {
        assert!(delta == 1 || delta == -1);
        let mut ranges: Vec<_> = ranges
            .iter()
            .copied()
            .filter(|&(_, bytes)| bytes != 0)
            .collect();
        if ranges.is_empty() {
            return Ok(());
        }
        ranges.sort_unstable_by_key(|&(address, _)| address);
        let mut coalesced: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (address, bytes) in ranges {
            let end = address
                .checked_add(bytes)
                .filter(|&end| end <= 1 << self.0.address_bits)
                .ok_or(MemoryError::Invalid("cached-page range exceeds SMMU"))?;
            let first = address & !(PAGE_SIZE - 1);
            let end = end.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if let Some((_, previous_end)) = coalesced.last_mut()
                && *previous_end >= first
            {
                *previous_end = (*previous_end).max(end);
            } else {
                coalesced.push((first, end));
            }
        }
        let _counter = self
            .0
            .counter_guard
            .lock(coalesced[0].0, coalesced.last().unwrap().1);
        for (first, end) in coalesced {
            self.update_pages_cached_count_no_lock(first, end, delta)?;
        }
        Ok(())
    }
    fn update_pages_cached_count_no_lock(
        &self,
        first: u64,
        end: u64,
        delta: i8,
    ) -> Result<(), MemoryError> {
        let state = self.0.state.read().unwrap();
        let mut pending: Option<(u32, u64, u64, bool)> = None;
        let mut callbacks = Vec::new();
        let mut flush = |pending: &mut Option<(u32, u64, u64, bool)>| -> Result<(), MemoryError> {
            if let Some((process, start, bytes, cached)) = pending.take()
                && let Some(space) = state.processes[process as usize].upgrade()
            {
                callbacks.push((space, start, bytes, cached));
            }
            Ok(())
        };
        for at in (first..end).step_by(PAGE_SIZE as usize) {
            let (cpu_process, cpu_address) = self.0.tracked.entry(at).cpu_backing();
            if cpu_address == 0 {
                flush(&mut pending)?;
                continue;
            }
            let counts = self.0.counts[(at >> LEAF_SHIFT) as usize]
                .get_or_init(|| Box::new(std::array::from_fn(|_| AtomicU8::new(0))));
            let count = &counts[((at >> 12) as usize) & (LEAF_PAGES - 1)];
            let previous = count.fetch_add(delta as u8, Ordering::Release);
            assert!(
                (delta > 0 && previous < u8::MAX) || (delta < 0 && previous > 0),
                "unbalanced SMMU cached-page references at {at:#x}: previous={previous}, delta={delta}, range={first:#x}..{end:#x}"
            );
            let changed = if delta > 0 {
                previous == 0
            } else {
                previous == 1
            };
            if !changed {
                flush(&mut pending)?;
                continue;
            }
            let cached = delta > 0;
            if let Some((process, start, size, state)) = &mut pending
                && *process == cpu_process
                && *start + *size == cpu_address
                && *state == cached
            {
                *size += PAGE_SIZE;
            } else {
                flush(&mut pending)?;
                pending = Some((cpu_process, cpu_address, PAGE_SIZE, cached));
            }
        }
        flush(&mut pending)?;
        drop(state);
        // As in the reference, cached counts select CPU process-memory callbacks.
        // A shared Rust map handle must release its metadata guard before those
        // callbacks publish page flags or inspect device aliases.
        for (space, start, bytes, _cached) in callbacks {
            space.refresh_cached_region(crate::GuestVirtualAddress::new(start), bytes, self)?;
        }
        Ok(())
    }
    pub(crate) fn physical_cached(&self, physical: crate::ArenaOffset) -> bool {
        let scratch = RefCell::new(Vec::new());
        let mut cached = false;
        self.apply_on_physical(
            crate::Extent {
                start: physical,
                bytes: 1,
            },
            &scratch,
            |address, _| {
                cached |= self.0.counts[(address >> LEAF_SHIFT) as usize]
                    .get()
                    .is_some_and(|counts| {
                        counts[((address >> 12) as usize) & (LEAF_PAGES - 1)]
                            .load(Ordering::Acquire)
                            != 0
                    });
            },
        );
        cached
    }
    /// Walk additional SMMU aliases using the same compressed reverse table as
    /// CPU write notifications. Called when cached GPU writes are invalidated,
    /// never as another admission step for each raw upload.
    pub fn apply_device_aliases(&self, address: u64, bytes: u64, mut apply: impl FnMut(u64, u64)) {
        let scratch = RefCell::new(Vec::new());
        let end = address + bytes;
        let mut at = address;
        while at < end {
            let count = (PAGE_SIZE - (at & (PAGE_SIZE - 1))).min(end - at);
            if let Some(physical) = self.physical_address(at) {
                self.apply_on_physical(
                    crate::Extent {
                        start: physical,
                        bytes: count,
                    },
                    &scratch,
                    |alias, size| {
                        if alias != at {
                            apply(alias, size);
                        }
                    },
                );
            }
            at += count;
        }
    }
    pub fn take_changes(&self) -> crate::RangeSet {
        std::mem::take(&mut self.0.state.write().unwrap().changed)
    }
    /// Reverse SMMU translation for physical CPU notifications. Every device
    /// alias receives the write, as DeviceMemoryManager::GetDeviceAddressesForNative.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
    pub fn device_ranges(&self, extent: crate::Extent) -> Vec<(u64, u64)> {
        let mut ranges = crate::RangeSet::default();
        let scratch = RefCell::new(Vec::new());
        self.apply_on_physical(extent, &scratch, |address, bytes| {
            ranges.insert(address, bytes)
        });
        ranges.iter().collect()
    }

    pub fn find_address(
        &self,
        requested: &PhysicalMapping,
        range: std::ops::Range<u64>,
    ) -> Result<Option<u64>, MemoryError> {
        let Some(span) = requested
            .spans()
            .iter()
            .find(|span| span.logical <= range.start && range.start < span.logical + span.bytes)
        else {
            return Ok(None);
        };
        let physical = span.start.get() + range.start - span.logical;
        let check = |page: u32| {
            let candidate = u64::from(page - 1) * PAGE_SIZE + (physical & (PAGE_SIZE - 1));
            let view = self.read_mapping(candidate, range.end - range.start)?;
            Ok((view.containing_offset(requested, range.clone()) == Some(0)).then_some(candidate))
        };
        let index = (physical / PAGE_SIZE) as usize;
        let head = self.0.reverse[index].load(Ordering::Acquire);
        if head & MULTIPLE == 0 {
            return if head == 0 { Ok(None) } else { check(head) };
        }
        let mut scratch = Vec::new();
        self.gather_device_pages(index, &mut scratch);
        for page in scratch {
            if let Some(candidate) = check(page)? {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }
    /// Native storage geometry is independent of the start of any later read.
    /// Actual uploads use ReadBlockUnsafe/WalkBlock on their consumed address range.
    pub fn buffer_mapping(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<Arc<PhysicalMapping>, MemoryError> {
        self.mapping(address, bytes, false)
    }
    /// ReadBlockUnsafe: advance by CPU physical continuity, and zero an entire
    /// block if its starting SMMU page is unmapped.
    pub fn unmapped_blocks(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<smallvec::SmallVec<[std::ops::Range<u64>; 4]>, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| end <= 1 << self.0.address_bits)
            .ok_or(MemoryError::Invalid(
                "device block range exceeds address space",
            ))?;
        let mut at = address;
        let mut holes: smallvec::SmallVec<[std::ops::Range<u64>; 4]> = smallvec::SmallVec::new();
        while at < end {
            let page = self.0.tracked.entry(at).page();
            let count = ((u64::from(page.continuity) + 1) * PAGE_SIZE - (at & (PAGE_SIZE - 1)))
                .min(end - at);
            if page.physical == 0 {
                if let Some(last) = holes.last_mut()
                    && last.end == at - address
                {
                    last.end += count;
                } else {
                    holes.push(at - address..at - address + count);
                }
            }
            at += count;
        }
        Ok(holes)
    }
    pub fn read_mapping(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<Arc<PhysicalMapping>, MemoryError> {
        self.mapping(address, bytes, true)
    }
    fn mapping(
        &self,
        address: u64,
        bytes: u64,
        read: bool,
    ) -> Result<Arc<PhysicalMapping>, MemoryError> {
        let end = address
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= 1 << self.0.address_bits)
            .ok_or(MemoryError::Invalid(
                "SMMU buffer range exceeds device address space",
            ))?;
        let state = self.0.state.read().unwrap();
        let mut parts = Vec::new();
        let mut at = address;
        while at < end {
            let page = self.0.tracked.entry(at).page();
            let within = at & (PAGE_SIZE - 1);
            let count = ((u64::from(page.continuity) + 1) * PAGE_SIZE - within).min(end - at);
            if page.physical == 0 {
                at += if read { count } else { PAGE_SIZE - within };
                continue;
            }
            let (base, backing) = state
                .sources
                .range(..=at)
                .next_back()
                .and_then(|(&base, backing)| backing.upgrade().map(|backing| (base, backing)))
                .filter(|(base, backing)| at >= *base && at + count <= *base + backing.bytes())
                .ok_or(MemoryError::Invalid("SMMU backing retired during read"))?;
            let mapping = backing.slice(at - base, count)?;
            if mapping.spans().first().is_none_or(|span| {
                span.start.get() != u64::from(page.physical - 1) * PAGE_SIZE + within
            }) {
                return Err(MemoryError::Invalid(
                    "SMMU continuity refers to changed CPU physical backing",
                ));
            }
            parts.push((at - address, mapping));
            at += count;
        }
        PhysicalMapping::sparse(self.machine().identity(), bytes, &parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_pages(
        machine: &Machine,
        address: u64,
        bytes: u64,
    ) -> (Arc<crate::AddressSpace>, Arc<[AllocationRange]>) {
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let space = crate::AddressSpace::new(machine.clone(), crate::AddressWidth::Bits32).unwrap();
        space
            .map(
                crate::GuestVirtualAddress::new(address),
                allocation,
                0,
                bytes,
                crate::MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let inputs = space
            .capture(
                crate::GuestVirtualAddress::new(address),
                bytes,
                crate::MemoryPermissions::READ_WRITE,
            )
            .unwrap()
            .into();
        (space, inputs)
    }

    #[test]
    fn shared_cached_page_counts_change_real_cpu_protection_only_at_zero_and_one() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let (space, input) = cpu_pages(&machine, 0x10000, 2 * PAGE_SIZE);
        let memory = DeviceMemory::new(machine.clone(), 34);
        let reservation = memory.reserve(2 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let base = reservation.address();
        let mapped = memory.map(base, input, true).unwrap();
        let before = machine.tracking_statistics().protection_calls;
        memory
            .update_pages_cached_count(base + 3, 2 * PAGE_SIZE - 6, 1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 1);
        memory
            .update_pages_cached_count(base, 2 * PAGE_SIZE, 1)
            .unwrap();
        memory
            .update_pages_cached_count(base, 2 * PAGE_SIZE, -1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 1);
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        assert!(
            !space
                .write_native(&native, crate::GuestVirtualAddress::new(0x10000), &[7])
                .unwrap()
        );
        drop(native);
        memory
            .update_pages_cached_count(base, 2 * PAGE_SIZE, -1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 2);
        let native = core.enter_native().unwrap();
        assert!(
            space
                .write_native(&native, crate::GuestVirtualAddress::new(0x11000), &[9])
                .unwrap()
        );
        drop(native);
        drop(mapped);
    }
    #[test]
    fn cached_counts_split_cpu_processes_and_unmapped_device_holes() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let (_first, first) = cpu_pages(&machine, 0x10000, 2 * PAGE_SIZE);
        let (_second, second) = cpu_pages(&machine, 0x30000, PAGE_SIZE);
        let memory = DeviceMemory::new(machine.clone(), 34);
        let reservation = memory.reserve(4 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let base = reservation.address();
        let _a = memory
            .map(
                base,
                vec![first[0].slice(0, PAGE_SIZE).unwrap()].into(),
                true,
            )
            .unwrap();
        let _b = memory
            .map(
                base + 2 * PAGE_SIZE,
                vec![first[0].slice(PAGE_SIZE, PAGE_SIZE).unwrap()].into(),
                true,
            )
            .unwrap();
        let _c = memory.map(base + 3 * PAGE_SIZE, second, true).unwrap();
        let before = machine.tracking_statistics().protection_calls;
        memory
            .update_pages_cached_count(base, 4 * PAGE_SIZE, 1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 3);
        memory
            .update_pages_cached_count(base, 4 * PAGE_SIZE, -1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 6);
        memory
            .update_pages_cached_count(base + PAGE_SIZE, PAGE_SIZE, 1)
            .unwrap();
        memory
            .update_pages_cached_count(base + PAGE_SIZE, PAGE_SIZE, -1)
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, before + 6);
    }
    #[test]
    fn mapped_page_reads_retain_ram_until_the_last_smmu_user_retires() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
        let input = AllocationRange::new(allocation.pin_range(0, PAGE_SIZE).unwrap(), 0, PAGE_SIZE)
            .unwrap();
        machine
            .write_resolved(
                std::slice::from_ref(&input),
                &vec![0x5a; PAGE_SIZE as usize],
            )
            .unwrap();
        let memory = DeviceMemory::new(machine.clone(), 34);
        let reader = memory.reader();
        let reservation = memory.reserve(2 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let address = reservation.address();
        let mapped = memory.map(address, vec![input].into(), true).unwrap();
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        let recycled = account.allocate(PAGE_SIZE).unwrap();
        machine
            .write_resolved(
                &[AllocationRange::new(recycled.pin(), 0, PAGE_SIZE).unwrap()],
                &vec![0xc7; PAGE_SIZE as usize],
            )
            .unwrap();
        let mut bytes = [0; 32];
        reader.read_block_unsafe(address + 17, &mut bytes).unwrap();
        assert_eq!(bytes, [0x5a; 32]);
        // A page-crossing read must retain WalkBlock's mapped/unmapped behavior.
        reader
            .read_block_unsafe(address + PAGE_SIZE - 1, &mut bytes[..2])
            .unwrap();
        assert_eq!(bytes[..2], [0x5a, 0]);
        drop(mapped);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        reader.read_block_unsafe(address, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 32]);
        drop(recycled);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn cache_geometry_and_read_start_remain_distinct_and_reverse_aliases_retire() {
        let machine = Machine::new(4 * 65536).unwrap();
        let allocation = machine.account(65536).unwrap().allocate(65536).unwrap();
        let full = AllocationRange::new(allocation.pin(), 0, 65536).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&full), &vec![0x5a; 65536])
            .unwrap();
        let backing = PhysicalMapping::new(std::slice::from_ref(&full)).unwrap();
        let memory = DeviceMemory::new(machine, 34);
        let reader = memory.reader();
        let reservation = memory.reserve(65536, 65536).unwrap();
        let base = reservation.address();
        memory.track_backing(base, &backing).unwrap();
        let page: Arc<[AllocationRange]> = vec![full.slice(0xb000, 4096).unwrap()].into();
        let mapped = memory.map(base + 0xb000, Arc::clone(&page), false).unwrap();
        let storage = memory.buffer_mapping(base, 65536).unwrap();
        assert_eq!(storage.spans()[0].logical, 0xb000);
        assert_eq!(
            memory.find_address(mapped.physical(), 0..32).unwrap(),
            Some(base + 0xb000)
        );
        let mut bytes = vec![0xff; 65536];
        reader.read_block_unsafe(base, &mut bytes).unwrap();
        assert!(
            bytes.iter().all(|&byte| byte == 0),
            "an unmapped start skips the whole CPU continuity run"
        );
        reader
            .read_block_unsafe(base + 0xb000, &mut bytes[..32])
            .unwrap();
        assert_eq!(&bytes[..32], &[0x5a; 32]);
        let alias_reservation = memory.reserve(65536, 65536).unwrap();
        let alias_address = alias_reservation.address();
        let alias = memory.map(alias_address, page, true).unwrap();
        let physical = mapped.physical().spans()[0].start.get();
        let changed = crate::Extent {
            start: crate::ArenaOffset::new(physical + 7),
            bytes: 13,
        };
        assert_eq!(
            memory.device_ranges(changed),
            [(base + 0xb000 + 7, 13), (alias_address + 7, 13)]
        );

        drop(mapped);
        assert_eq!(memory.device_ranges(changed), [(alias_address + 7, 13)]);
        assert_eq!(
            memory.find_address(alias.physical(), 0..32).unwrap(),
            Some(alias_address)
        );
        assert!(memory.take_changes().contains(base + 0xb000, 4096));
        assert!(memory.take_changes().is_empty());
        let physical = Arc::clone(alias.physical());
        drop(alias);
        assert_eq!(memory.find_address(&physical, 0..32).unwrap(), None);
    }
}

#[cfg(test)]
mod reader_tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn retracked_cpu_holes_do_not_keep_previous_backing_continuity() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let full = AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap();
        machine
            .write_resolved(
                std::slice::from_ref(&full),
                &vec![0x75; full.bytes() as usize],
            )
            .unwrap();
        let backing = PhysicalMapping::new(std::slice::from_ref(&full)).unwrap();
        let tail = full.slice(2 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let sparse = PhysicalMapping::sparse(
            machine.identity(),
            3 * PAGE_SIZE,
            &[(
                2 * PAGE_SIZE,
                PhysicalMapping::new(std::slice::from_ref(&tail)).unwrap(),
            )],
        )
        .unwrap();
        let memory = DeviceMemory::new(machine, 34);
        let reservation = memory.reserve(3 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let address = reservation.address();
        let reader = memory.reader();
        memory.track_backing(address, &backing).unwrap();
        let _mapped = memory
            .map(address + 2 * PAGE_SIZE, vec![tail].into(), false)
            .unwrap();
        let mut output = vec![0xff; 3 * PAGE_SIZE as usize];
        reader.read_block_unsafe(address, &mut output).unwrap();
        assert!(output.iter().all(|&byte| byte == 0));

        memory.track_backing(address, &sparse).unwrap();
        reader.read_block_unsafe(address, &mut output).unwrap();
        assert!(
            output[..2 * PAGE_SIZE as usize]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(
            output[2 * PAGE_SIZE as usize..]
                .iter()
                .all(|&byte| byte == 0x75)
        );
    }

    #[test]
    fn live_device_readers_invalidate_all_four_slots_after_unmap_and_remap() {
        let machine = Machine::new(10 * PAGE_SIZE).unwrap();
        let account = machine.account(10 * PAGE_SIZE).unwrap();
        let old = account.allocate(5 * PAGE_SIZE).unwrap();
        let new = account.allocate(5 * PAGE_SIZE).unwrap();
        let old_range = AllocationRange::new(old.pin(), 0, 5 * PAGE_SIZE).unwrap();
        let new_range = AllocationRange::new(new.pin(), 0, 5 * PAGE_SIZE).unwrap();
        let before: Vec<_> = (0..5)
            .flat_map(|i| vec![0x30 + i; PAGE_SIZE as usize])
            .collect();
        let after: Vec<_> = (0..5)
            .flat_map(|i| vec![0xb0 + i; PAGE_SIZE as usize])
            .collect();
        machine
            .write_resolved(std::slice::from_ref(&old_range), &before)
            .unwrap();
        machine
            .write_resolved(std::slice::from_ref(&new_range), &after)
            .unwrap();
        let memory = DeviceMemory::new(machine, 34);
        let reservation = memory.reserve(5 * PAGE_SIZE, PAGE_SIZE).unwrap();
        let address = reservation.address();
        let mapped = memory.map(address, vec![old_range].into(), true).unwrap();
        let reader = memory.reader();
        let other = memory.reader();
        let (commands, input) = mpsc::channel();
        let (output, results) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            for page in input {
                let mut value = [0];
                other
                    .read_block_unsafe(address + page * PAGE_SIZE + 17, &mut value)
                    .unwrap();
                output.send(value[0]).unwrap();
            }
        });
        for page in 0..4 {
            let mut value = [0];
            reader
                .read_block_unsafe(address + page * PAGE_SIZE + 17, &mut value)
                .unwrap();
            assert_eq!(value[0], 0x30 + page as u8);
            commands.send(page).unwrap();
            assert_eq!(results.recv().unwrap(), value[0]);
        }
        drop(mapped);
        for page in 0..4 {
            let mut value = [0xff];
            reader
                .read_block_unsafe(address + page * PAGE_SIZE + 17, &mut value)
                .unwrap();
            assert_eq!(value, [0]);
            commands.send(page).unwrap();
            assert_eq!(results.recv().unwrap(), 0);
        }
        let _mapped = memory.map(address, vec![new_range].into(), true).unwrap();
        for page in [0, 1, 2, 3, 4, 0] {
            let mut value = [0];
            reader
                .read_block_unsafe(address + page * PAGE_SIZE + 17, &mut value)
                .unwrap();
            assert_eq!(value[0], 0xb0 + page as u8);
            commands.send(page).unwrap();
            assert_eq!(results.recv().unwrap(), value[0]);
        }
        let mut crossing = [0; 16];
        reader
            .read_block_unsafe(address + PAGE_SIZE - 8, &mut crossing)
            .unwrap();
        assert_eq!(&crossing[..8], &[0xb0; 8]);
        assert_eq!(&crossing[8..], &[0xb1; 8]);
        drop(commands);
        worker.join().unwrap();
    }
}
