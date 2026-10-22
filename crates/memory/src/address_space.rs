//! Indexed process translations; mapping records retain allocations, not pages.
mod alias;
use crate::transition::{CoreContext, NativeAdmission, Quiescence};
use crate::{
    Access, AccessOperation, AddressSpaceId, AddressWidth, Allocation, AllocationId, AllocationPin,
    ArenaOffset, Extent, GuestVirtualAddress, Machine, MemoryError, MemoryPermissions, PAGE_SIZE,
    VirtualArena,
};
use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering},
    },
};

/// Unpacked local view of one atomic entry; never stored as a second table.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct MappedPage {
    pub monitor_flags: u64,
    pub physical_page: u32,
    pub permissions: MemoryPermissions,
    pub kind: u8,
}
impl MappedPage {
    pub(crate) const RAM: u8 = 1;
    pub(crate) const OBSERVED: u8 = 2;
    pub(crate) const DEVICE: u8 = 4;
    const EMPTY: Self = Self {
        monitor_flags: 0,
        physical_page: 0,
        permissions: MemoryPermissions::NONE,
        kind: 0,
    };
    fn present(self) -> bool {
        self.kind & Self::RAM != 0
    }
    fn require(
        self,
        address: GuestVirtualAddress,
        required: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        if !self.present() {
            return Err(MemoryError::Unmapped(address));
        }
        if !self.permissions.contains(required) {
            return Err(MemoryError::PermissionDenied { address, required });
        }
        Ok(())
    }
    fn direct_permissions(self) -> MemoryPermissions {
        let permissions = if self.kind & Self::DEVICE != 0 {
            MemoryPermissions::NONE
        } else if self.kind & Self::OBSERVED != 0 {
            self.permissions
                .intersection(MemoryPermissions::READ_EXECUTE)
        } else {
            self.permissions
        };
        if permissions.contains(MemoryPermissions::READ) {
            permissions.intersection(MemoryPermissions::READ_WRITE)
        } else {
            MemoryPermissions::NONE
        }
    }
}
/// One atomic tagged pointer per guest page. Ownership and mapping versions
/// live in mapping records, outside the native access table.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/page_table.h
#[repr(transparent)]
pub struct PageEntry(AtomicU64);
const _: () = assert!(std::mem::size_of::<PageEntry>() == 8);
const TRACKED: u64 = 1;
const DEVICE: u64 = 2;
const RESERVED: u64 = 4;
const PERMISSION_SHIFT: u32 = 3;
const ATTRIBUTES: u64 = 63;
#[repr(transparent)]
struct Leaf([PageEntry; 512]);
struct PageTableStorage {
    machine: Machine,
    base: std::ptr::NonNull<Leaf>,
    bytes: usize,
}
impl PageTableStorage {
    fn new(machine: &Machine, width: AddressWidth) -> Result<Self, MemoryError> {
        let bytes = (width.bytes() / PAGE_SIZE) as usize * std::mem::size_of::<PageEntry>();
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        machine.reserve_page_table(bytes as u64);
        let table = Self {
            machine: machine.clone(),
            base: std::ptr::NonNull::new(base.cast()).expect("nonzero page-table reservation"),
            bytes,
        };
        if unsafe { libc::madvise(base, bytes, libc::MADV_DONTDUMP) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // A tiny initialized leaf must not commit an entire 2 MiB huge page.
        unsafe {
            libc::madvise(base, bytes, libc::MADV_NOHUGEPAGE);
        }
        Ok(table)
    }
    fn leaf(&self, index: usize) -> *mut Leaf {
        self.base.as_ptr().wrapping_add(index)
    }
}
impl Drop for PageTableStorage {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.bytes);
        }
        self.machine.release_page_table(self.bytes as u64);
    }
}
struct Mapping {
    generation: u64,
    bytes: u64,
    allocation: Allocation,
    offset: u64,
    aliases: Vec<crate::IntervalKey>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Translation {
    pub physical: ArenaOffset,
    pub allocation: AllocationId,
    pub generation: u64,
    pub permissions: MemoryPermissions,
}
/// Source translation of a captured range. Physical operations retain this
/// version as evidence; they never use its address to retranslate old inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappingVersion {
    pub address_space: AddressSpaceId,
    pub start: GuestVirtualAddress,
    pub generation: u64,
}
pub enum CodeFetch {
    Ready {
        word: u32,
    },
    Capture {
        ranges: Vec<AllocationRange>,
        device_owned: bool,
    },
}
#[derive(Clone)]
pub struct MappingObservation {
    _registration: Option<Arc<MappingRegistration>>,
}
struct MappingRegistration {
    space: Weak<AddressSpace>,
    key: crate::IntervalKey,
}
impl Drop for MappingRegistration {
    fn drop(&mut self) {
        if let Some(space) = self.space.upgrade() {
            space.mapping_observers.lock().unwrap().remove(self.key);
        }
    }
}
/// Immutable physical capture. Retention is per allocation/range, never per page.
#[derive(Clone)]
pub struct AllocationRange {
    allocation: AllocationPin,
    offset: u64,
    bytes: u64,
    mapping: Option<MappingVersion>,
}
impl std::fmt::Debug for AllocationRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationRange")
            .field("machine", &self.allocation.machine().identity())
            .field("allocation", &self.allocation.identity())
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .field("mapping", &self.mapping)
            .finish()
    }
}
impl AllocationRange {
    pub fn new(allocation: AllocationPin, offset: u64, bytes: u64) -> Result<Self, MemoryError> {
        if !allocation.covers(offset, bytes) {
            return Err(MemoryError::Invalid(
                "physical capture exceeds its retained allocation",
            ));
        }
        Ok(Self {
            allocation,
            offset,
            bytes,
            mapping: None,
        })
    }
    pub fn allocation(&self) -> &AllocationPin {
        &self.allocation
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn mapping(&self) -> Option<MappingVersion> {
        self.mapping
    }

    /// Retain only this logical subrange, including its original VA generation.
    /// The new lease does not keep unrelated pages from the parent capture alive.
    pub fn slice(&self, offset: u64, bytes: u64) -> Result<Self, MemoryError> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(MemoryError::Invalid(
                "subcapture exceeds its original physical input",
            ));
        }
        let allocation_offset = self.offset + offset;
        Ok(Self {
            allocation: self.allocation.pin_range(allocation_offset, bytes)?,
            offset: allocation_offset,
            bytes,
            mapping: self.mapping.map(|mut mapping| {
                mapping.start = mapping
                    .start
                    .checked_add(offset)
                    .expect("captured virtual range is bounded");
                mapping
            }),
        })
    }

    #[inline]
    pub fn physical_extents(&self) -> impl Iterator<Item = Extent> + '_ {
        self.allocation.physical_extents(self.offset, self.bytes)
    }
}

/// Select ordered bytes from an already accepted input/output mapping. No VA
/// lookup, permission recheck or later guest-memory read occurs here.
pub fn capture_subrange(
    ranges: &[AllocationRange],
    offset: u64,
    bytes: u64,
) -> Result<Vec<AllocationRange>, MemoryError> {
    let total = ranges
        .iter()
        .try_fold(0u64, |total, range| total.checked_add(range.bytes))
        .ok_or(MemoryError::Invalid("captured input length overflows"))?;
    let end = offset
        .checked_add(bytes)
        .filter(|end| *end <= total)
        .ok_or(MemoryError::Invalid(
            "subcapture exceeds its original physical input",
        ))?;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let mut selected = Vec::new();
    let mut cursor = 0;
    for range in ranges {
        let next = cursor + range.bytes;
        if offset < next && end > cursor {
            let start = offset.saturating_sub(cursor);
            let stop = range.bytes.min(end - cursor);
            selected.push(range.slice(start, stop - start)?);
        }
        cursor = next;
        if cursor >= end {
            break;
        }
    }
    Ok(selected)
}
pub struct AddressSpace {
    id: AddressSpaceId,
    arena: VirtualArena,
    pub(crate) quiescence: Quiescence,
    table: PageTableStorage,
    root: Box<[AtomicPtr<Leaf>]>,
    mappings: UnsafeCell<BTreeMap<u64, Mapping>>,
    next_generation: AtomicU64,
    self_weak: std::sync::Weak<AddressSpace>,
    mapping_observers: Mutex<crate::IntervalIndex<Weak<AtomicBool>>>,
}
// SAFETY: leaves persist until destruction; entries atomically publish pointer
// and attributes. Mapping-record changes require native/host exclusion. Data
// tracking instead excludes short checked accesses and protects all aliases.
// Core contexts retain the address space until every native admission ends.
unsafe impl Send for AddressSpace {}
unsafe impl Sync for AddressSpace {}
impl AddressSpace {
    pub fn new(machine: Machine, width: AddressWidth) -> Result<Arc<Self>, MemoryError> {
        static NEXT_SPACE: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_SPACE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("address-space identity exhausted"))?;
        let table = PageTableStorage::new(&machine, width)?;
        let arena = VirtualArena::new(machine, width)?;
        Ok(Arc::new_cyclic(|weak| Self {
            self_weak: weak.clone(),
            id: AddressSpaceId::new(id),
            arena,
            quiescence: Quiescence::default(),
            table,
            root: (0..1usize << (width.bits() - 21))
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            mappings: UnsafeCell::new(BTreeMap::new()),
            next_generation: AtomicU64::new(1),
            mapping_observers: Mutex::new(crate::IntervalIndex::default()),
        }))
    }
    pub fn identity(&self) -> AddressSpaceId {
        self.id
    }
    pub fn machine(&self) -> &Machine {
        self.arena.machine()
    }
    pub fn width(&self) -> AddressWidth {
        self.arena.width()
    }
    pub fn native_page_table_base(&self) -> *const PageEntry {
        self.table.base.as_ptr().cast()
    }
    pub fn native_data_base(&self) -> crate::HostAddress {
        self.arena
            .host_address(GuestVirtualAddress::new(0), 0)
            .expect("live virtual reservation")
    }
    pub fn register_core(self: &Arc<Self>, interrupt: Arc<dyn Fn() + Send + Sync>) -> CoreContext {
        self.quiescence.register(Arc::clone(self), interrupt)
    }
    /// Guest instruction-cache maintenance changes compiled code, without
    /// changing physical RAM, permissions or mapping generations.
    pub fn invalidate_code(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MemoryError> {
        self.range(start, bytes)?;
        let transition = self
            .quiescence
            .begin_transition(self.quiescence.writer()?)?;
        transition.wait_quiescent();
        self.quiescence.invalidate_code(start, bytes);
        transition.publish();
        Ok(())
    }
    /// Make a terminal host/device failure visible to every core and access.
    pub fn fail(&self, error: MemoryError) -> MemoryError {
        self.quiescence.fail_execution(error)
    }
    /// End guest access even while stopped cores retain this address-space
    /// owner. Physical captures remain independent and may finish on their old
    /// allocation pins. Neither cached metadata nor stopped cores retain RAM.
    pub fn close(&self) -> Result<(), MemoryError> {
        let mut aliases = self.machine().aliases();
        let Some(transition) = self.quiescence.begin_close()? else {
            return Ok(());
        };
        transition.wait_quiescent();
        unsafe {
            self.arena
                .unmap(GuestVirtualAddress::new(0), self.width().bytes())
        }
        .map_err(|error| transition.fail(error))?;
        let removed = std::mem::take(unsafe { &mut *self.mappings.get() });
        for mapping in removed.values() {
            for &key in &mapping.aliases {
                aliases.remove(key);
            }
        }
        for root in &self.root {
            let leaf = root.load(Ordering::Relaxed);
            if !leaf.is_null() {
                unsafe {
                    for entry in &(*leaf).0 {
                        entry.0.store(0, Ordering::Release);
                    }
                }
            }
        }
        self.invalidate_mapping_observers(GuestVirtualAddress::new(0), self.width().bytes());
        self.quiescence.finish_close();
        transition.publish();
        drop(aliases);
        drop(removed);
        Ok(())
    }
    fn range(&self, start: GuestVirtualAddress, bytes: u64) -> Result<u64, MemoryError> {
        let end = start
            .get()
            .checked_add(bytes)
            .filter(|&end| bytes != 0 && end <= self.width().bytes())
            .ok_or(MemoryError::AddressRange { start, bytes })?;
        Ok(end)
    }
    fn pages(&self, start: GuestVirtualAddress, bytes: u64) -> Result<u64, MemoryError> {
        let end = self.range(start, bytes)?;
        if !start.get().is_multiple_of(PAGE_SIZE) || !bytes.is_multiple_of(PAGE_SIZE) {
            return Err(MemoryError::Invalid(
                "mapping must cover whole virtual pages",
            ));
        }
        Ok(end)
    }
    // Admission retains mappings and any activated physical monitor. Tracking
    // updates replace pointer and attributes in one aligned atomic operation.
    unsafe fn entry(&self, address: u64) -> MappedPage {
        let Some(root) = self.root.get((address >> 21) as usize) else {
            return MappedPage::EMPTY;
        };
        let leaf = root.load(Ordering::Acquire);
        if leaf.is_null() {
            return MappedPage::EMPTY;
        }
        let raw = unsafe {
            (*leaf).0[((address >> 12) & 511) as usize]
                .0
                .load(Ordering::Acquire)
        };
        if raw == 0 {
            return MappedPage::EMPTY;
        }
        let pointer = raw & !ATTRIBUTES;
        let canonical = if raw & RESERVED != 0 {
            unsafe { &*(pointer as *const crate::reservation::PhysicalMonitor) }.canonical_base()
        } else {
            pointer as usize
        };
        let base = self
            .machine()
            .arena()
            .host_address(ArenaOffset::new(0), 0)
            .unwrap()
            .as_ptr() as usize;
        MappedPage {
            monitor_flags: if raw & RESERVED != 0 { pointer } else { 1 },
            physical_page: ((canonical - base) / PAGE_SIZE as usize) as u32,
            permissions: MemoryPermissions::from_bits(((raw >> PERMISSION_SHIFT) & 7) as u8)
                .unwrap(),
            kind: MappedPage::RAM
                | if raw & TRACKED != 0 {
                    MappedPage::OBSERVED
                } else {
                    0
                }
                | if raw & DEVICE != 0 {
                    MappedPage::DEVICE
                } else {
                    0
                },
        }
    }
    fn encode_entry(&self, entry: MappedPage) -> u64 {
        if !entry.present() {
            return 0;
        }
        let reserved = entry.monitor_flags > 1;
        let pointer = if reserved {
            entry.monitor_flags
        } else {
            self.machine()
                .arena()
                .host_address(
                    ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE),
                    PAGE_SIZE,
                )
                .unwrap()
                .as_ptr() as u64
        };
        debug_assert_eq!(pointer & ATTRIBUTES, 0);
        pointer
            | (u64::from(entry.permissions.bits()) << PERMISSION_SHIFT)
            | if entry.kind & MappedPage::OBSERVED != 0 {
                TRACKED
            } else {
                0
            }
            | if entry.kind & MappedPage::DEVICE != 0 {
                DEVICE
            } else {
                0
            }
            | if reserved { RESERVED } else { 0 }
    }
    // New mappings are installed under native/host exclusion. Existing entries
    // may change tracking under the mapping writer while native cores continue.
    unsafe fn set_entry(&self, address: u64, entry: MappedPage) {
        let root = &self.root[(address >> 21) as usize];
        let mut leaf = root.load(Ordering::Relaxed);
        if leaf.is_null() {
            leaf = self.table.leaf((address >> 21) as usize);
            root.store(leaf, Ordering::Release);
        }
        unsafe {
            (*leaf).0[((address >> 12) & 511) as usize]
                .0
                .store(self.encode_entry(entry), Ordering::Release);
        }
    }
    // Caller holds the machine alias index and this space's writer, and has
    // drained native/host accesses. Physical code watches are still armed.
    pub(crate) unsafe fn invalidate_code_writes(&self, start: u64, bytes: u64) {
        let end = start + bytes;
        let mut run = None;
        for address in (start..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            let executable = entry.present()
                && entry.permissions.contains(MemoryPermissions::EXECUTE)
                && self
                    .machine()
                    .tracking()
                    .code(u64::from(entry.physical_page));
            match (run, executable) {
                (None, true) => run = Some(address),
                (Some(begin), false) => {
                    self.quiescence
                        .invalidate_code(GuestVirtualAddress::new(begin), address - begin);
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = run {
            self.quiescence
                .invalidate_code(GuestVirtualAddress::new(begin), end - begin);
        }
    }
    pub fn translate(
        &self,
        address: GuestVirtualAddress,
        required: MemoryPermissions,
    ) -> Result<Translation, MemoryError> {
        self.range(address, 1)?;
        let _access = self.quiescence.enter_host()?;
        unsafe { self.translate_admitted(address, required) }
    }
    pub fn check_native_access(
        &self,
        native: &NativeAdmission<'_>,
        address: GuestVirtualAddress,
        required: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        if !std::ptr::eq(self, native.address_space()) || !native.is_active() {
            return Err(MemoryError::Invalid(
                "native access requires active admission for this address space",
            ));
        }
        self.range(address, 1)?;
        unsafe { self.entry(address.get()) }.require(address, required)
    }
    /// A compilation callback must return Capture to its caller, leave C++ and
    /// suspend admission before establishing new physical code tracking.
    pub fn fetch_code_native(
        &self,
        native: &NativeAdmission<'_>,
        address: GuestVirtualAddress,
    ) -> Result<CodeFetch, MemoryError> {
        if !address.get().is_multiple_of(4) {
            return Err(MemoryError::Invalid("unaligned A64 instruction fetch"));
        }
        self.range(address, 4)?;
        self.check_native_access(native, address, MemoryPermissions::EXECUTE)?;
        let entry = unsafe { self.entry(address.get()) };
        if entry.kind & MappedPage::DEVICE == 0
            && self
                .machine()
                .tracking()
                .code(u64::from(entry.physical_page))
        {
            let source = self
                .machine()
                .arena()
                .host_address(
                    ArenaOffset::new(
                        u64::from(entry.physical_page) * PAGE_SIZE + address.get() % PAGE_SIZE,
                    ),
                    4,
                )
                .expect("admitted code input");
            return Ok(CodeFetch::Ready {
                word: unsafe { crate::raw::read_value(source.as_ptr(), 4) } as u32,
            });
        }
        unsafe {
            self.capture_admitted(address.get(), address.get() + 4, MemoryPermissions::EXECUTE)
        }
        .map(|ranges| CodeFetch::Capture {
            ranges,
            device_owned: entry.kind & MappedPage::DEVICE != 0,
        })
    }
    unsafe fn translate_admitted(
        &self,
        address: GuestVirtualAddress,
        required: MemoryPermissions,
    ) -> Result<Translation, MemoryError> {
        let entry = unsafe { self.entry(address.get()) };
        entry.require(address, required)?;
        let mappings = unsafe { &*self.mappings.get() };
        let (_, mapping) = mappings
            .range(..=address.get())
            .next_back()
            .expect("present mapping metadata");
        Ok(Translation {
            physical: ArenaOffset::new(
                u64::from(entry.physical_page) * PAGE_SIZE + address.get() % PAGE_SIZE,
            ),
            allocation: mapping.allocation.identity(),
            generation: mapping.generation,
            permissions: entry.permissions,
        })
    }
    pub fn map(
        &self,
        start: GuestVirtualAddress,
        allocation: Allocation,
        offset: u64,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        self.map_impl(start, allocation, offset, bytes, permissions, false)
    }
    pub fn remap(
        &self,
        start: GuestVirtualAddress,
        allocation: Allocation,
        offset: u64,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        self.map_impl(start, allocation, offset, bytes, permissions, true)
    }
    fn map_impl(
        &self,
        start: GuestVirtualAddress,
        allocation: Allocation,
        offset: u64,
        bytes: u64,
        permissions: MemoryPermissions,
        replace: bool,
    ) -> Result<(), MemoryError> {
        let end = self.pages(start, bytes)?;
        let pin = allocation.pin();
        if pin.machine().identity() != self.machine().identity()
            || !offset.is_multiple_of(PAGE_SIZE)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > allocation.bytes())
            || !allocation.owns(offset, bytes)
        {
            return Err(MemoryError::Invalid(
                "mapping names a foreign or invalid physical allocation range",
            ));
        }
        let allocation = allocation.view(offset, bytes)?;
        let mut aliases = self.machine().aliases();
        let writer = self.quiescence.writer()?;
        let mut code_changed = permissions.contains(MemoryPermissions::EXECUTE);
        for address in (start.get()..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            let present = entry.present();
            code_changed |= entry.permissions.contains(MemoryPermissions::EXECUTE);
            if present != replace {
                return Err(MemoryError::Invalid(if replace {
                    "replacement crosses an unmapped virtual page"
                } else {
                    "virtual mapping overlaps an existing mapping"
                }));
            }
        }
        let generation = self
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("mapping identity exhausted"))?;
        let physical = AllocationRange {
            allocation: pin,
            offset,
            bytes,
            mapping: None,
        };
        let transition = self.quiescence.begin_transition(writer)?;
        transition.wait_quiescent();
        unsafe { self.map_physical_entries(start, &physical, permissions) }
            .map_err(|error| transition.fail(error))?;
        let removed = if replace {
            unsafe { self.remove_mappings(&mut aliases, start.get(), end) }
                .map_err(|error| transition.fail(error))?
        } else {
            Vec::new()
        };
        let keys = self
            .register_aliases(&mut aliases, start.get(), &physical)
            .map_err(|error| transition.fail(error))?;
        unsafe { &mut *self.mappings.get() }.insert(
            start.get(),
            Mapping {
                generation,
                bytes,
                allocation,
                offset,
                aliases: keys,
            },
        );
        if code_changed {
            self.quiescence.invalidate_code(start, bytes);
        }
        self.invalidate_mapping_observers(start, bytes);
        transition.publish();
        drop(aliases);
        drop(removed);
        Ok(())
    }
    // Caller holds alias/writer serialization and has drained native/host
    // admission. The retained physical input covers whole validated pages.
    unsafe fn map_physical_entries(
        &self,
        start: GuestVirtualAddress,
        physical: &AllocationRange,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let mut address = start.get();
        for extent in physical.physical_extents() {
            let mut run = None;
            for at in (0..extent.bytes).step_by(PAGE_SIZE as usize) {
                let page = (extent.start.get() + at) / PAGE_SIZE;
                let page = u32::try_from(page).map_err(|_| {
                    MemoryError::Invalid("physical page exceeds translation entry capacity")
                })?;
                let entry = MappedPage {
                    monitor_flags: self.machine().tracking().native_flags(u64::from(page)),
                    kind: self.machine().tracking().kind(u64::from(page)),
                    physical_page: page,
                    permissions,
                };
                let direct = entry.direct_permissions();
                if let Some((base, previous)) = run.filter(|(_, previous)| *previous != direct) {
                    unsafe {
                        self.arena.map(
                            GuestVirtualAddress::new(address + base),
                            ArenaOffset::new(extent.start.get() + base),
                            at - base,
                            previous,
                        )
                    }?;
                    run = None;
                }
                if run.is_none() {
                    run = Some((at, direct));
                }
                unsafe {
                    self.set_entry(address + at, entry);
                }
            }
            if let Some((base, direct)) = run {
                unsafe {
                    self.arena.map(
                        GuestVirtualAddress::new(address + base),
                        ArenaOffset::new(extent.start.get() + base),
                        extent.bytes - base,
                        direct,
                    )
                }?;
            }
            address += extent.bytes;
        }
        Ok(())
    }
    pub fn capture(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        required: MemoryPermissions,
    ) -> Result<Vec<AllocationRange>, MemoryError> {
        let end = self.range(start, bytes)?;
        let _access = self.quiescence.enter_host()?;
        unsafe { self.capture_admitted(start.get(), end, required) }
    }
    /// Pin a naturally aligned scalar together with its accepted access rights.
    /// Kernel operations can keep these rights across an asynchronous access
    /// without translating a potentially replaced virtual mapping again.
    pub fn capture_scalar(
        &self,
        start: GuestVirtualAddress,
        bytes: u8,
        required: MemoryPermissions,
    ) -> Result<(AllocationRange, MemoryPermissions), MemoryError> {
        let bytes = u64::from(bytes);
        if !matches!(bytes, 1 | 2 | 4 | 8 | 16) || !start.get().is_multiple_of(bytes) {
            return Err(MemoryError::Invalid(
                "captured scalar has invalid width or alignment",
            ));
        }
        let end = self.range(start, bytes)?;
        let _access = self.quiescence.enter_host()?;
        let permissions = unsafe { self.entry(start.get()) }.permissions;
        let mut ranges = unsafe { self.capture_admitted(start.get(), end, required) }?;
        debug_assert_eq!(ranges.len(), 1);
        Ok((
            ranges.pop().expect("one naturally aligned scalar"),
            permissions,
        ))
    }
    /// Subscribe a cached physical capture's existing invalid bit to its source
    /// mapping. Validate captured versions under admission before registration;
    /// a remap between byte capture and this call invalidates the cache instead
    /// of attaching an old snapshot to the new VA. This retains neither AS nor RAM.
    pub fn observe_mapping(
        &self,
        range: &AllocationRange,
        observation: &crate::Observation,
    ) -> Result<MappingObservation, MemoryError> {
        let source = range
            .mapping()
            .filter(|source| source.address_space == self.identity())
            .ok_or(MemoryError::Invalid(
                "mapping observation requires a virtual capture from this address space",
            ))?;
        let end = self.range(source.start, range.bytes())?;
        let _access = self.quiescence.enter_host()?;
        let mut valid = !observation.is_invalid();
        let mappings = unsafe { &*self.mappings.get() };
        let mut address = source.start.get();
        while address < end {
            let Some((&base, mapping)) = mappings
                .range(..=address)
                .next_back()
                .filter(|(base, mapping)| **base + mapping.bytes > address)
            else {
                valid = false;
                break;
            };
            valid &= mapping.generation == source.generation
                && mapping.allocation.identity() == range.allocation().identity();
            address = (base + mapping.bytes).min(end);
        }
        if !valid {
            observation.invalidate();
            return Ok(MappingObservation {
                _registration: None,
            });
        }
        let key = self.mapping_observers.lock().unwrap().insert(
            source.start.get(),
            end,
            observation.weak_invalidation_flag(),
        )?;
        Ok(MappingObservation {
            _registration: Some(Arc::new(MappingRegistration {
                space: self.self_weak.clone(),
                key,
            })),
        })
    }
    fn invalidate_mapping_observers(&self, start: GuestVirtualAddress, bytes: u64) {
        let mut observers = self.mapping_observers.lock().unwrap();
        let mut invalidated = Vec::new();
        observers.overlapping_with_keys(start.get(), start.get() + bytes, |key, flag| {
            if let Some(flag) = flag.upgrade() {
                flag.store(true, Ordering::Release);
            }
            invalidated.push(key);
        });
        for key in invalidated {
            observers.remove(key);
        }
    }
    pub fn mapping_observation_count(&self) -> usize {
        self.mapping_observers.lock().unwrap().len()
    }
    pub fn capture_native(
        &self,
        native: &NativeAdmission<'_>,
        start: GuestVirtualAddress,
        bytes: u64,
        required: MemoryPermissions,
    ) -> Result<Vec<AllocationRange>, MemoryError> {
        if !std::ptr::eq(self, native.address_space()) || !native.is_active() {
            return Err(MemoryError::Invalid(
                "native translation requires active admission for this address space",
            ));
        }
        let end = self.range(start, bytes)?;
        unsafe { self.capture_admitted(start.get(), end, required) }
    }
    unsafe fn capture_admitted(
        &self,
        start: u64,
        end: u64,
        required: MemoryPermissions,
    ) -> Result<Vec<AllocationRange>, MemoryError> {
        for address in (start & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            entry.require(GuestVirtualAddress::new(address.max(start)), required)?;
        }
        let mappings = unsafe { &*self.mappings.get() };
        let mut address = start;
        let mut ranges = Vec::new();
        while address < end {
            let (&base, mapping) = mappings
                .range(..=address)
                .next_back()
                .expect("present page retains its allocation");
            let stop = (base + mapping.bytes).min(end);
            ranges.push(AllocationRange {
                allocation: mapping
                    .allocation
                    .pin_range(mapping.offset + address - base, stop - address)?,
                offset: mapping.offset + address - base,
                bytes: stop - address,
                mapping: Some(MappingVersion {
                    address_space: self.identity(),
                    start: GuestVirtualAddress::new(address),
                    generation: mapping.generation,
                }),
            });
            address = stop;
        }
        Ok(ranges)
    }
    fn validate_access(
        &self,
        start: u64,
        end: u64,
        required: MemoryPermissions,
    ) -> Result<u8, MemoryError> {
        let mut kinds = 0;
        for address in (start & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            entry.require(GuestVirtualAddress::new(address.max(start)), required)?;
            kinds |= entry.kind;
        }
        Ok(kinds)
    }
    pub fn read_into(
        &self,
        start: GuestVirtualAddress,
        destination: &mut [u8],
    ) -> Result<Access<()>, MemoryError> {
        if destination.is_empty() {
            return Ok(Access::Ready(()));
        }
        let end = self.range(start, destination.len() as u64)?;
        let access = self.quiescence.enter_host()?;
        let kinds = self.validate_access(start.get(), end, MemoryPermissions::READ)?;
        if kinds & MappedPage::DEVICE != 0 {
            let ranges =
                unsafe { self.capture_admitted(start.get(), end, MemoryPermissions::READ) }?;
            drop(access);
            return self
                .machine()
                .request_device_access(ranges, AccessOperation::Read)
                .map(Access::Pending);
        }
        let mut address = start.get();
        let mut written = 0;
        while address < end {
            let entry = unsafe { self.entry(address) };
            let offset = address % PAGE_SIZE;
            let bytes = (PAGE_SIZE - offset).min(end - address) as usize;
            let physical = ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE + offset);
            let source = self
                .machine()
                .arena()
                .host_address(physical, bytes as u64)
                .expect("mapped physical range");
            unsafe {
                crate::raw::capture(source.as_ptr(), &mut destination[written..written + bytes]);
            }
            address += bytes as u64;
            written += bytes;
        }
        Ok(Access::Ready(()))
    }
    pub fn write(
        &self,
        start: GuestVirtualAddress,
        source: &[u8],
    ) -> Result<Access<()>, MemoryError> {
        if source.is_empty() {
            return Ok(Access::Ready(()));
        }
        let end = self.range(start, source.len() as u64)?;
        let access = self.quiescence.enter_host()?;
        // Validate the full operation before publishing its first byte.
        let kinds = self.validate_access(start.get(), end, MemoryPermissions::WRITE)?;
        if kinds & MappedPage::DEVICE != 0 {
            let ranges =
                unsafe { self.capture_admitted(start.get(), end, MemoryPermissions::WRITE) }?;
            drop(access);
            return self
                .machine()
                .request_device_access(ranges, AccessOperation::Write(source.into()))
                .map(Access::Pending);
        }
        if kinds & MappedPage::OBSERVED != 0 {
            let ranges =
                unsafe { self.capture_admitted(start.get(), end, MemoryPermissions::WRITE) }?;
            drop(access);
            return self
                .machine()
                .write_captured(&crate::CapturedMemory::new(ranges)?, 0, source);
        }
        // SAFETY: host admission retains all mappings and excludes physical
        // tracking transitions; the complete write has been validated above.
        unsafe { self.write_admitted(start.get(), end, source) };
        Ok(Access::Ready(()))
    }
    /// Complete ordinary stores or collect device-dependent overwrites without
    /// leaving native execution. Observation invalidation and collection
    /// contention defer without writing. Validate cross-page stores in full.
    pub fn write_native(
        &self,
        native: &NativeAdmission<'_>,
        start: GuestVirtualAddress,
        source: &[u8],
    ) -> Result<bool, MemoryError> {
        if !std::ptr::eq(self, native.address_space()) || !native.is_active() {
            return Err(MemoryError::Invalid(
                "native write requires active admission for this address space",
            ));
        }
        if source.is_empty() {
            return Ok(true);
        }
        let end = self.range(start, source.len() as u64)?;
        if source.len() > 16 {
            return Ok(false);
        }
        let mut guards = [None, None];
        for (index, page) in (start.get() & !(PAGE_SIZE - 1)..end)
            .step_by(PAGE_SIZE as usize)
            .enumerate()
        {
            let entry = unsafe { self.entry(page) };
            entry.require(
                GuestVirtualAddress::new(page.max(start.get())),
                MemoryPermissions::WRITE,
            )?;
            let extent = Extent {
                start: ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE),
                bytes: PAGE_SIZE,
            };
            let Some(guard) = self.machine().physical_access().try_enter(extent) else {
                return Ok(false);
            };
            guards[index] = Some(guard);
        }
        let kinds = self.validate_access(start.get(), end, MemoryPermissions::WRITE)?;
        if kinds & MappedPage::OBSERVED != 0 {
            return Ok(false);
        }
        if kinds & MappedPage::DEVICE != 0 {
            return native.cpu_writes().write(
                self.machine(),
                start.get(),
                source.len(),
                |page| {
                    ArenaOffset::new(
                        u64::from(unsafe { self.entry(page) }.physical_page) * PAGE_SIZE,
                    )
                },
                |page| {
                    let mut ranges = unsafe {
                        self.capture_admitted(page, page + PAGE_SIZE, MemoryPermissions::WRITE)
                    }?;
                    if ranges.len() != 1 {
                        return Err(MemoryError::Invalid(
                            "native dirty page has split physical identity",
                        ));
                    }
                    Ok(ranges.remove(0))
                },
                || unsafe { self.write_admitted(start.get(), end, source) },
            );
        }
        // SAFETY: native admission retains the mappings; short physical guards
        // exclude tracking changes. Contention defers outside Run, never waits
        // inside the native invocation on the writer draining its accesses.
        unsafe { self.write_admitted(start.get(), end, source) };
        Ok(true)
    }
    /// Caller holds native/host admission and has validated all target pages
    /// as writable, unobserved RAM. Device-dependent stores additionally hold
    /// the CPU-write collection exclusion through their exact dirty marking.
    unsafe fn write_admitted(&self, start: u64, end: u64, source: &[u8]) {
        let mut address = start;
        let mut consumed = 0;
        while address < end {
            let entry = unsafe { self.entry(address) };
            let offset = address % PAGE_SIZE;
            let bytes = (PAGE_SIZE - offset).min(end - address) as usize;
            let physical = ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE + offset);
            unsafe {
                self.machine().write_admitted(
                    Extent {
                        start: physical,
                        bytes: bytes as u64,
                    },
                    &source[consumed..consumed + bytes],
                );
            }
            address += bytes as u64;
            consumed += bytes;
        }
    }
    pub fn read_value(
        &self,
        start: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<Access<u128>, MemoryError> {
        if !matches!(bytes, 1 | 2 | 4 | 8 | 16) {
            return Err(MemoryError::Invalid("unsupported scalar memory width"));
        }
        self.range(start, bytes as u64)?;
        if start.get() % PAGE_SIZE + bytes as u64 > PAGE_SIZE {
            let mut value = [0; 16];
            return match self.read_into(start, &mut value[..bytes])? {
                Access::Ready(()) => Ok(Access::Ready(u128::from_le_bytes(value))),
                Access::Pending(pending) => Ok(Access::Pending(pending)),
            };
        }
        let access = self.quiescence.enter_host()?;
        let entry = unsafe { self.entry(start.get()) };
        entry.require(start, MemoryPermissions::READ)?;
        let physical =
            ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE + start.get() % PAGE_SIZE);
        if entry.kind & MappedPage::DEVICE != 0 {
            let ranges = unsafe {
                self.capture_admitted(
                    start.get(),
                    start.get() + bytes as u64,
                    MemoryPermissions::READ,
                )
            }?;
            drop(access);
            return self
                .machine()
                .request_device_access(ranges, AccessOperation::Read)
                .map(Access::Pending);
        }
        let source = self
            .machine()
            .arena()
            .host_address(physical, bytes as u64)
            .expect("mapped physical scalar");
        Ok(Access::Ready(unsafe {
            crate::raw::read_value(source.as_ptr(), bytes)
        }))
    }
    /// Complete a native read without suspending the current admission. A
    /// resolved area permits RAM reads inside a mixed CPU/device page; it owns
    /// no guest bytes and is checked against the current physical translation.
    pub fn read_native(
        &self,
        native: &NativeAdmission<'_>,
        start: GuestVirtualAddress,
        bytes: usize,
        area: Option<&crate::NativeReadArea>,
    ) -> Result<Option<u128>, MemoryError> {
        if !std::ptr::eq(self, native.address_space()) || !native.is_active() {
            return Err(MemoryError::Invalid(
                "native read requires active admission for this address space",
            ));
        }
        if !matches!(bytes, 1 | 2 | 4 | 8 | 16) {
            return Err(MemoryError::Invalid("unsupported native read width"));
        }
        self.range(start, bytes as u64)?;
        if start.get() % PAGE_SIZE + bytes as u64 > PAGE_SIZE {
            return Ok(None);
        }
        let entry = unsafe { self.entry(start.get()) };
        entry.require(start, MemoryPermissions::READ)?;
        let physical = u64::from(entry.physical_page) * PAGE_SIZE + start.get() % PAGE_SIZE;
        let Some(_access) = self.machine().physical_access().try_enter(Extent {
            start: ArenaOffset::new(physical),
            bytes: bytes as u64,
        }) else {
            return Ok(None);
        };
        let entry = unsafe { self.entry(start.get()) };
        if entry.kind & MappedPage::DEVICE != 0
            && !area.is_some_and(|area| {
                area.machine == self.machine().identity()
                    && !area.observation.is_invalid()
                    && physical >= area.start.get()
                    && physical + bytes as u64 <= area.start.get() + area.bytes
            })
        {
            return Ok(None);
        }
        let source = self
            .machine()
            .arena()
            .host_address(ArenaOffset::new(physical), bytes as u64)
            .expect("admitted physical native read");
        // Admission pins the translation; physical exclusion prevents GPU
        // invalidation of this proof. CPU stores use the same raw RAM protocol;
        // every value is loaded afresh rather than cached.
        Ok(Some(unsafe {
            crate::raw::read_value(source.as_ptr(), bytes)
        }))
    }
    pub fn write_value(
        &self,
        start: GuestVirtualAddress,
        bytes: usize,
        value: u128,
    ) -> Result<Access<()>, MemoryError> {
        if !matches!(bytes, 1 | 2 | 4 | 8 | 16) {
            return Err(MemoryError::Invalid("unsupported scalar memory width"));
        }
        self.range(start, bytes as u64)?;
        if start.get() % PAGE_SIZE + bytes as u64 > PAGE_SIZE {
            return self.write(start, &value.to_le_bytes()[..bytes]);
        }
        let access = self.quiescence.enter_host()?;
        let entry = unsafe { self.entry(start.get()) };
        entry.require(start, MemoryPermissions::WRITE)?;
        let physical =
            ArenaOffset::new(u64::from(entry.physical_page) * PAGE_SIZE + start.get() % PAGE_SIZE);
        if entry.kind & MappedPage::DEVICE != 0 {
            let ranges = unsafe {
                self.capture_admitted(
                    start.get(),
                    start.get() + bytes as u64,
                    MemoryPermissions::WRITE,
                )
            }?;
            drop(access);
            return self
                .machine()
                .request_device_access(
                    ranges,
                    AccessOperation::Write(value.to_le_bytes()[..bytes].into()),
                )
                .map(Access::Pending);
        }
        if entry.kind & MappedPage::OBSERVED != 0 {
            let ranges = unsafe {
                self.capture_admitted(
                    start.get(),
                    start.get() + bytes as u64,
                    MemoryPermissions::WRITE,
                )
            }?;
            drop(access);
            return self.machine().write_captured(
                &crate::CapturedMemory::new(ranges)?,
                0,
                &value.to_le_bytes()[..bytes],
            );
        }
        let destination = self
            .machine()
            .arena()
            .host_address(physical, bytes as u64)
            .expect("mapped physical scalar");
        unsafe {
            crate::raw::write_value(destination.as_ptr(), bytes, value);
        }
        Ok(Access::Ready(()))
    }
    /// A host/kernel atomic captures its physical operation before leaving
    /// translation admission. Native callbacks use capture_native instead.
    pub fn atomic(
        &self,
        start: GuestVirtualAddress,
        operation: crate::AtomicAccess,
    ) -> Result<Access<crate::AccessResult>, MemoryError> {
        let bytes = u64::from(operation.bytes);
        if !matches!(bytes, 1 | 2 | 4 | 8 | 16) || !start.get().is_multiple_of(bytes) {
            return Err(MemoryError::Invalid(
                "atomic access has invalid width or alignment",
            ));
        }
        let required = match operation.operation {
            crate::AtomicOperation::LoadExclusive => MemoryPermissions::READ,
            crate::AtomicOperation::StoreExclusive { .. } => MemoryPermissions::WRITE,
            _ => MemoryPermissions::READ_WRITE,
        };
        let ranges = self.capture(start, bytes, required)?;
        self.machine()
            .access_captured(&ranges, AccessOperation::Atomic(operation))
    }
    pub fn protect(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let end = self.pages(start, bytes)?;
        let mut aliases = self.machine().aliases();
        let writer = self.quiescence.writer()?;
        let mut changed = false;
        let mut code_changed = false;
        for address in (start.get()..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            if !entry.present() {
                return Err(MemoryError::Invalid(
                    "permission change crosses an unmapped virtual page",
                ));
            }
            changed |= entry.permissions != permissions;
            code_changed |= entry.permissions.contains(MemoryPermissions::EXECUTE)
                || permissions.contains(MemoryPermissions::EXECUTE);
        }
        if !changed {
            return Ok(());
        }
        let generation = self
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("mapping identity exhausted"))?;
        let transition = self.quiescence.begin_transition(writer)?;
        transition.wait_quiescent();
        unsafe {
            self.update_page_permissions(start.get(), bytes, |mut entry| {
                entry.permissions = permissions;
                entry
            })
        }
        .map_err(|error| transition.fail(error))?;
        unsafe { self.update_mapping_generations(&mut aliases, start.get(), end, generation) }
            .map_err(|error| transition.fail(error))?;
        if code_changed {
            self.quiescence.invalidate_code(start, bytes);
        }
        self.invalidate_mapping_observers(start, bytes);
        transition.publish();
        Ok(())
    }
    pub fn unmap(&self, start: GuestVirtualAddress, bytes: u64) -> Result<(), MemoryError> {
        let end = self.pages(start, bytes)?;
        let mut aliases = self.machine().aliases();
        let writer = self.quiescence.writer()?;
        let mut code_changed = false;
        for address in (start.get()..end).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(address) };
            if !entry.present() {
                return Err(MemoryError::Invalid(
                    "unmap crosses an unmapped virtual page",
                ));
            }
            code_changed |= entry.permissions.contains(MemoryPermissions::EXECUTE);
        }
        let transition = self.quiescence.begin_transition(writer)?;
        transition.wait_quiescent();
        unsafe { self.arena.unmap(start, bytes) }.map_err(|error| transition.fail(error))?;
        for address in (start.get()..end).step_by(PAGE_SIZE as usize) {
            unsafe {
                self.set_entry(address, MappedPage::EMPTY);
            }
        }
        let removed = unsafe { self.remove_mappings(&mut aliases, start.get(), end) }
            .map_err(|error| transition.fail(error))?;
        if code_changed {
            self.quiescence.invalidate_code(start, bytes);
        }
        self.invalidate_mapping_observers(start, bytes);
        transition.publish();
        drop(aliases);
        // Last guest ownership release and physical reclamation run after access
        // publication, with no writer/allocator locks nested around callbacks.
        drop(removed);
        Ok(())
    }
    fn register_aliases(
        &self,
        aliases: &mut crate::aliases::Aliases,
        start: u64,
        range: &AllocationRange,
    ) -> Result<Vec<crate::IntervalKey>, MemoryError> {
        let mut address = start;
        let mut keys = Vec::new();
        for extent in range.physical_extents() {
            keys.push(aliases.insert(crate::aliases::PhysicalAlias {
                space: self.self_weak.clone(),
                physical: extent.start.get(),
                virtual_address: address,
                bytes: extent.bytes,
                allocation: range.allocation.identity(),
            })?);
            address += extent.bytes;
        }
        Ok(keys)
    }
    // The single atomic store keeps pointer and attributes coherent.
    unsafe fn publish_tracking(&self, address: u64, entry: MappedPage) {
        unsafe {
            self.set_entry(address, entry);
        }
    }
    fn tracked_entry(&self, mut entry: MappedPage) -> MappedPage {
        let tracking = self.machine().tracking();
        entry.monitor_flags = tracking.native_flags(u64::from(entry.physical_page));
        entry.kind = tracking.kind(u64::from(entry.physical_page));
        entry
    }
    // Host and checked physical accesses have drained. Tightening permissions
    // marks callbacks before protecting aliases; kernel protection drains old
    // native accesses before capture. Relaxation opens the kernel mapping
    // before any checked native load/store can observe direct RAM again.
    pub(crate) unsafe fn refresh_tracking(
        &self,
        start: u64,
        bytes: u64,
    ) -> Result<(), MemoryError> {
        let mut run: Option<(u64, MemoryPermissions, bool)> = None;
        for address in (start..start + bytes).step_by(PAGE_SIZE as usize) {
            let old = unsafe { self.entry(address) };
            let entry = self.tracked_entry(old);
            let permissions = entry.direct_permissions();
            let changed = old.direct_permissions() != permissions;
            let relaxing = !old.direct_permissions().contains(permissions);
            if let Some((base, previous, was_relaxing)) =
                run.filter(|(_, previous, was_relaxing)| {
                    !changed || *previous != permissions || *was_relaxing != relaxing
                })
            {
                unsafe {
                    self.protect_tracking_run(base, address - base, previous, was_relaxing)?;
                }
                run = None;
            }
            if !changed || !relaxing {
                unsafe {
                    self.publish_tracking(address, entry);
                }
            }
            if changed && run.is_none() {
                run = Some((address, permissions, relaxing));
            }
        }
        if let Some((base, permissions, relaxing)) = run {
            unsafe {
                self.protect_tracking_run(base, start + bytes - base, permissions, relaxing)?;
            }
        }
        Ok(())
    }
    unsafe fn protect_tracking_run(
        &self,
        start: u64,
        bytes: u64,
        permissions: MemoryPermissions,
        relaxing: bool,
    ) -> Result<(), MemoryError> {
        unsafe {
            self.arena
                .protect(GuestVirtualAddress::new(start), bytes, permissions)?;
        }
        self.machine().tracking().record_protection();
        if relaxing {
            for address in (start..start + bytes).step_by(PAGE_SIZE as usize) {
                let entry = self.tracked_entry(unsafe { self.entry(address) });
                unsafe {
                    self.publish_tracking(address, entry);
                }
            }
        }
        Ok(())
    }
    // Caller has quiesced this address space. Guest and tracking permission
    // changes share the same coalescing walk, changing only actual host access.
    unsafe fn update_page_permissions(
        &self,
        start: u64,
        bytes: u64,
        mut update: impl FnMut(MappedPage) -> MappedPage,
    ) -> Result<(), MemoryError> {
        let mut run = None;
        for address in (start..start + bytes).step_by(PAGE_SIZE as usize) {
            let old = unsafe { self.entry(address) };
            debug_assert!(old.present());
            let entry = update(old);
            let permissions = entry.direct_permissions();
            let changed = old.direct_permissions() != permissions;
            if let Some((base, previous)) =
                run.filter(|(_, previous)| !changed || *previous != permissions)
            {
                unsafe {
                    self.arena
                        .protect(GuestVirtualAddress::new(base), address - base, previous)?;
                }
                self.machine().tracking().record_protection();
                run = None;
            }
            if entry != old {
                unsafe {
                    self.set_entry(address, entry);
                }
            }
            if changed && run.is_none() {
                run = Some((address, permissions));
            }
        }
        if let Some((base, permissions)) = run {
            unsafe {
                self.arena.protect(
                    GuestVirtualAddress::new(base),
                    start + bytes - base,
                    permissions,
                )?;
            }
            self.machine().tracking().record_protection();
        }
        Ok(())
    }
    unsafe fn update_mapping_generations(
        &self,
        aliases: &mut crate::aliases::Aliases,
        start: u64,
        end: u64,
        generation: u64,
    ) -> Result<(), MemoryError> {
        let mut replacements = Vec::new();
        let mut address = start;
        while address < end {
            let mappings = unsafe { &*self.mappings.get() };
            let (&base, mapping) = mappings
                .range(..=address)
                .next_back()
                .expect("validated mapping range");
            let stop = (base + mapping.bytes).min(end);
            let offset = mapping.offset + address - base;
            replacements.push((
                address,
                Mapping {
                    generation,
                    bytes: stop - address,
                    allocation: mapping.allocation.view(offset, stop - address)?,
                    offset,
                    aliases: Vec::new(),
                },
            ));
            address = stop;
        }
        // New views retain every original page before any old view is dropped.
        let removed = unsafe { self.remove_mappings(aliases, start, end)? };
        for (address, mut mapping) in replacements {
            let range =
                AllocationRange::new(mapping.allocation.pin(), mapping.offset, mapping.bytes)?;
            mapping.aliases = self.register_aliases(aliases, address, &range)?;
            unsafe { &mut *self.mappings.get() }.insert(address, mapping);
        }
        drop(removed);
        Ok(())
    }
    unsafe fn remove_mappings(
        &self,
        aliases: &mut crate::aliases::Aliases,
        start: u64,
        end: u64,
    ) -> Result<Vec<Mapping>, MemoryError> {
        let mappings = unsafe { &mut *self.mappings.get() };
        let mut removed = Vec::new();
        loop {
            let next = mappings
                .range(..end)
                .next_back()
                .map(|(&base, map)| (base, base + map.bytes));
            let Some((base, stop)) = next.filter(|&(_, stop)| stop > start) else {
                break;
            };
            let mapping = mappings.remove(&base).unwrap();
            for &key in &mapping.aliases {
                aliases.remove(key);
            }
            if base < start {
                mappings.insert(
                    base,
                    Mapping {
                        generation: mapping.generation,
                        bytes: start - base,
                        allocation: mapping.allocation.view(mapping.offset, start - base)?,
                        offset: mapping.offset,
                        aliases: self.register_aliases(
                            aliases,
                            base,
                            &AllocationRange {
                                allocation: mapping.allocation.pin(),
                                offset: mapping.offset,
                                bytes: start - base,
                                mapping: None,
                            },
                        )?,
                    },
                );
            }
            if stop > end {
                mappings.insert(
                    end,
                    Mapping {
                        generation: mapping.generation,
                        bytes: stop - end,
                        allocation: mapping
                            .allocation
                            .view(mapping.offset + end - base, stop - end)?,
                        offset: mapping.offset + end - base,
                        aliases: self.register_aliases(
                            aliases,
                            end,
                            &AllocationRange {
                                allocation: mapping.allocation.pin(),
                                offset: mapping.offset + end - base,
                                bytes: stop - end,
                                mapping: None,
                            },
                        )?,
                    },
                );
            }
            removed.push(mapping);
        }
        Ok(removed)
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        self.invalidate_mapping_observers(GuestVirtualAddress::new(0), self.width().bytes());
        // No native context can outlive this space. Remove reverse mappings
        // before field destruction unmaps the arena and releases guest owners.
        let mut aliases = self.machine().aliases();
        for mapping in unsafe { &*self.mappings.get() }.values() {
            for &key in &mapping.aliases {
                aliases.remove(key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_callback_never_waits_for_the_code_capture_draining_its_core() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 4).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let (interrupt, interrupted) = std::sync::mpsc::channel();
        let context = space.register_core(Arc::new(move || {
            interrupt.send(()).unwrap();
        }));
        let native = context.enter_native().unwrap();
        let observing = std::thread::spawn(move || {
            let observation = crate::Observation::default();
            let result = machine
                .access_captured(
                    &[input],
                    crate::AccessOperation::ReadCode(observation.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            (observation, result)
        });
        interrupted
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        // The halt notification requests quiescence; the still-admitted callback
        // may finish before physical closure, or defer after it. It must never
        // wait for its own native admission to drain.
        let wrote = space.write_native(&native, base, &[1, 2, 3, 4]).unwrap();
        native.suspend();
        let (observed, result) = observing.join().unwrap();
        assert_eq!(
            result,
            crate::AccessResult::Read(Box::new(if wrote { [1, 2, 3, 4] } else { [0; 4] }))
        );
        assert!(!observed.is_invalid());
        native.resume().unwrap();
        assert!(!space.write_native(&native, base, &[5, 6, 7, 8]).unwrap());
        drop(native);
        space
            .write_value(base, 4, 0x08070605)
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(observed.is_invalid());
    }
    #[test]
    fn native_writes_validate_all_pages_and_defer_tracking_without_partial_stores() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        let second = GuestVirtualAddress::new(2 * PAGE_SIZE);
        space
            .map(
                base,
                allocation,
                0,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
            .protect(second, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        let context = space.register_core(Arc::new(|| panic!("callback must not quiesce itself")));
        let native = context.enter_native().unwrap();
        let start = GuestVirtualAddress::new(2 * PAGE_SIZE - 8);
        assert!(space.write_native(&native, start, &[7; 16]).is_err());
        native.suspend();
        assert!(matches!(
            space.read_value(start, 8).unwrap(),
            Access::Ready(0)
        ));
        space
            .protect(second, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        let inputs = space
            .capture(start, 16, MemoryPermissions::READ_WRITE)
            .unwrap();
        let reserved = inputs[0].slice(0, 8).unwrap();
        let (_, proof) = machine.load_exclusive_resolved(&reserved).unwrap();
        native.resume().unwrap();
        assert!(space.write_native(&native, start, &[7; 16]).unwrap());
        native.suspend();
        assert!(
            !machine
                .store_exclusive_resolved(&reserved, proof, 9)
                .unwrap()
        );
        assert!(
            matches!(space.read_value(start, 16).unwrap(), Access::Ready(value) if value == u128::from_le_bytes([7; 16]))
        );
        let observation = crate::Observation::default();
        machine
            .access_captured(
                &inputs,
                crate::AccessOperation::ReadObserved(observation.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        native.resume().unwrap();
        assert!(!space.write_native(&native, start, &[8; 16]).unwrap());
        assert!(!observation.is_invalid());
        native.suspend();
        machine.publish_device_write(&inputs, || Ok(())).unwrap();
        native.resume().unwrap();
        assert!(!space.write_native(&native, start, &[9; 16]).unwrap());
        native.suspend();
        let mut bytes = [0; 16];
        machine.capture_stream(&inputs, &mut bytes).unwrap();
        assert_eq!(bytes, [7; 16]);
    }
    #[test]
    fn captured_scalar_keeps_its_original_rights_and_physical_word_after_remapping() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let original = account.allocate(PAGE_SIZE).unwrap();
        let original_id = original.identity();
        let initial = AllocationRange::new(original.pin(), 12, 4).unwrap();
        machine.write_resolved(&[initial], &[1, 2, 3, 4]).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(base, original, 0, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        let address = base.checked_add(12).unwrap();
        assert!(
            space
                .capture_scalar(address, 4, MemoryPermissions::READ_WRITE)
                .is_err()
        );
        assert!(
            space
                .capture_scalar(base.checked_add(13).unwrap(), 4, MemoryPermissions::READ)
                .is_err()
        );
        let (word, permissions) = space
            .capture_scalar(address, 4, MemoryPermissions::READ)
            .unwrap();
        space.unmap(base, PAGE_SIZE).unwrap();
        let replacement = account.allocate(PAGE_SIZE).unwrap();
        space
            .map(
                base,
                replacement,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        assert_eq!(permissions, MemoryPermissions::READ);
        assert_eq!(word.allocation().identity(), original_id);
        let mut bytes = [0; 4];
        assert!(matches!(
            machine
                .read_captured_into(&word.into(), 0, &mut bytes)
                .unwrap(),
            crate::Access::Ready(())
        ));
        assert_eq!(bytes, [1, 2, 3, 4]);
    }
    #[test]
    fn mapping_observers_validate_capture_versions_and_do_not_retain_owners() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let first = GuestVirtualAddress::new(PAGE_SIZE);
        let second = GuestVirtualAddress::new(2 * PAGE_SIZE);
        let mut owners = Vec::new();
        for address in [first, second] {
            let allocation = account.allocate(PAGE_SIZE).unwrap();
            owners.push(allocation.clone());
            space
                .map(
                    address,
                    allocation,
                    0,
                    PAGE_SIZE,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
        }
        let captures = [
            space.capture(first, 4, MemoryPermissions::READ).unwrap(),
            space.capture(second, 4, MemoryPermissions::READ).unwrap(),
        ];
        let first_observed = crate::Observation::default();
        machine
            .access_captured(
                &captures[0],
                crate::AccessOperation::ReadObserved(first_observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let second_observed = crate::Observation::default();
        machine
            .access_captured(
                &captures[1],
                crate::AccessOperation::ReadObserved(second_observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let first_watch = space
            .observe_mapping(&captures[0][0], &first_observed)
            .unwrap();
        let first_clone = first_watch.clone();
        let second_watch = space
            .observe_mapping(&captures[1][0], &second_observed)
            .unwrap();
        assert_eq!(space.mapping_observation_count(), 2);
        drop(first_watch);
        assert_eq!(space.mapping_observation_count(), 2);
        space
            .protect(first, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        assert!(!first_observed.is_invalid());
        space
            .protect(first, PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        assert!(first_observed.is_invalid());
        assert!(!second_observed.is_invalid());
        assert_eq!(space.mapping_observation_count(), 1);
        drop(first_clone);
        assert_eq!(space.mapping_observation_count(), 1);

        let late = crate::Observation::default();
        machine
            .access_captured(
                &captures[1],
                crate::AccessOperation::ReadObserved(late.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        space
            .remap(
                second,
                account.allocate(PAGE_SIZE).unwrap(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        assert!(second_observed.is_invalid());
        assert!(!late.is_invalid()); // Its physical allocation still has a guest owner.
        let late_watch = space.observe_mapping(&captures[1][0], &late).unwrap();
        assert!(late.is_invalid());
        assert_eq!(space.mapping_observation_count(), 0);
        let current = space.capture(second, 4, MemoryPermissions::READ).unwrap();
        let current_observed = crate::Observation::default();
        machine
            .access_captured(
                &current,
                crate::AccessOperation::ReadObserved(current_observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let current_watch = space
            .observe_mapping(&current[0], &current_observed)
            .unwrap();
        drop(second_watch); // An old token cannot remove a reused interval slot.
        assert_eq!(space.mapping_observation_count(), 1);
        drop(captures);
        drop(current);
        drop(owners);
        drop(space);
        assert!(current_observed.is_invalid());
        assert_eq!(machine.statistics().allocated_bytes, 0);
        drop((late_watch, current_watch));
    }
    #[test]
    fn native_code_capture_leaves_admission_and_tracks_executable_alias_writes() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let initial = AllocationRange::new(allocation.pin(), 0, 4).unwrap();
        machine
            .write_resolved(&[initial], &0x11223344u32.to_le_bytes())
            .unwrap();
        let code = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let data = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let pc = GuestVirtualAddress::new(0x1000);
        let alias = GuestVirtualAddress::new(0x4000);
        code.map(
            pc,
            allocation.clone(),
            0,
            PAGE_SIZE,
            MemoryPermissions::EXECUTE,
        )
        .unwrap();
        data.map(
            alias,
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
        let context = code.register_core(Arc::new(|| {
            panic!("this core is outside native execution during transitions")
        }));
        let native = context.enter_native().unwrap();
        let CodeFetch::Capture {
            ranges: captured, ..
        } = code.fetch_code_native(&native, pc).unwrap()
        else {
            panic!("unwatched code bypassed capture");
        };
        let version = captured[0].mapping().unwrap().generation;
        native.suspend();
        let (access, observation) = machine.access_code_captured(captured).unwrap();
        assert_eq!(
            access.into_ready().unwrap(),
            crate::AccessResult::Read(Box::new(0x11223344u32.to_le_bytes()))
        );
        assert!(!observation.is_invalid());
        native.resume().unwrap();
        let CodeFetch::Ready { word } = code.fetch_code_native(&native, pc).unwrap() else {
            panic!("tracked CPU code was not reusable");
        };
        assert_eq!(word, 0x11223344);
        assert_eq!(
            code.capture_native(&native, pc, 4, MemoryPermissions::EXECUTE)
                .unwrap()[0]
                .mapping()
                .unwrap()
                .generation,
            version
        );
        assert!(
            code.fetch_code_native(&native, GuestVirtualAddress::new(0x1001))
                .is_err()
        );
        native.suspend();
        assert!(
            code.read_value(pc, 4).is_err(),
            "execute permission must not grant data reads"
        );
        let atomic = crate::AtomicAccess {
            bytes: 4,
            ordering: crate::AccessOrdering::AcquireRelease,
            operation: crate::AtomicOperation::CompareExchange {
                expected: 0x11223344,
                replacement: 0xaabbccdd,
            },
        };
        assert!(
            code.atomic(pc, atomic).is_err(),
            "code permission must not grant atomic writes"
        );
        assert_eq!(
            data.atomic(alias, atomic).unwrap().into_ready().unwrap(),
            crate::AccessResult::Atomic {
                previous: 0x11223344,
                written: true
            }
        );
        assert!(observation.is_invalid());
        native.resume().unwrap();
        let mut invalidated = Vec::new();
        native
            .synchronize_code(|range| {
                invalidated.push(range);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            invalidated,
            [crate::CodeInvalidation {
                start: pc,
                bytes: PAGE_SIZE
            }]
        );
        assert!(matches!(
            code.fetch_code_native(&native, pc).unwrap(),
            CodeFetch::Capture { .. }
        ));
    }
    #[test]
    fn captured_versions_split_permission_changes_without_changing_physical_inputs() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let id = allocation.identity();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(
                base,
                allocation,
                0,
                3 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let original = space
            .capture(base, 3 * PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(original.len(), 1);
        let version = original[0].mapping().unwrap();
        assert_eq!(version.address_space, space.identity());
        assert_eq!(version.start, base);
        let observed = crate::Observation::default();
        machine
            .access_captured(
                &original,
                crate::AccessOperation::ReadObserved(observed.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!observed.is_invalid());
        let tracked = space
            .capture(base, 3 * PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(tracked.len(), 1);
        assert_eq!(tracked[0].mapping(), Some(version));
        space
            .protect(
                GuestVirtualAddress::new(0x2000),
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let start = GuestVirtualAddress::new(0x1ffe);
        let spans = space
            .capture(start, PAGE_SIZE + 4, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(
            spans.iter().map(AllocationRange::bytes).collect::<Vec<_>>(),
            [2, PAGE_SIZE, 2]
        );
        for (range, offset, address) in [
            (&spans[0], PAGE_SIZE - 2, start),
            (&spans[1], PAGE_SIZE, GuestVirtualAddress::new(0x2000)),
            (&spans[2], 2 * PAGE_SIZE, GuestVirtualAddress::new(0x3000)),
        ] {
            assert_eq!(range.allocation().identity(), id);
            assert_eq!(range.offset(), offset);
            assert_eq!(range.mapping().unwrap().start, address);
        }
        assert_eq!(spans[0].mapping().unwrap().generation, version.generation);
        assert_ne!(spans[1].mapping().unwrap().generation, version.generation);
        assert_eq!(spans[2].mapping().unwrap().generation, version.generation);
        assert_eq!(original[0].mapping(), Some(version));
        assert!(
            space
                .capture(start, PAGE_SIZE + 4, MemoryPermissions::WRITE)
                .is_err()
        );
        let context = space.register_core(Arc::new(|| {}));
        let native = context.enter_native().unwrap();
        let current = space
            .capture_native(
                &native,
                GuestVirtualAddress::new(0x2000),
                4,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(
            current[0].mapping().unwrap().generation,
            spans[1].mapping().unwrap().generation
        );
        assert_eq!(
            current[0].physical_extents().next().unwrap().start,
            spans[1].physical_extents().next().unwrap().start
        );
        native.suspend();
        assert!(
            space
                .check_native_access(&native, base, MemoryPermissions::READ)
                .is_err()
        );
        assert!(
            space
                .capture_native(&native, base, 4, MemoryPermissions::READ)
                .is_err()
        );
    }
    #[test]
    fn scalar_and_bulk_operations_cover_boundaries_without_partial_forbidden_writes() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x3000);
        space
            .map(
                base,
                allocation,
                0,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        for width in [1, 2, 4, 8, 16] {
            for displacement in [0, 1, 7, PAGE_SIZE - 3, PAGE_SIZE + 1] {
                let address = base.checked_add(displacement).unwrap();
                let value = 0x0123456789abcdef_fedcba9876543210u128;
                space
                    .write_value(address, width, value)
                    .unwrap()
                    .into_ready()
                    .unwrap();
                let mask = u128::MAX >> (128 - width * 8);
                assert_eq!(
                    space
                        .read_value(address, width)
                        .unwrap()
                        .into_ready()
                        .unwrap(),
                    value & mask
                );
            }
        }
        let bytes: Vec<u8> = (0..PAGE_SIZE + 33).map(|n| (n * 31) as u8).collect();
        space
            .write(base.checked_add(9).unwrap(), &bytes)
            .unwrap()
            .into_ready()
            .unwrap();
        let mut read = vec![0; bytes.len()];
        space
            .read_into(base.checked_add(9).unwrap(), &mut read)
            .unwrap()
            .into_ready()
            .unwrap();
        assert_eq!(bytes, read);
        space
            .unmap(base.checked_add(PAGE_SIZE).unwrap(), PAGE_SIZE)
            .unwrap();
        let end = base.checked_add(PAGE_SIZE - 2).unwrap();
        let original = space.read_value(end, 2).unwrap().into_ready().unwrap();
        assert!(space.write_value(end, 4, 0).is_err());
        assert_eq!(
            space.read_value(end, 2).unwrap().into_ready().unwrap(),
            original
        );
    }
    #[test]
    fn concurrent_mixed_width_alias_accesses_do_not_create_guest_rust_references() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let a = GuestVirtualAddress::new(0x1000);
        let b = GuestVirtualAddress::new(0x2000);
        space
            .map(
                a,
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
            .map(b, allocation, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        std::thread::scope(|threads| {
            for (lane, value) in [(0, 0u128), (1, u128::MAX)] {
                let space = Arc::clone(&space);
                threads.spawn(move || {
                    for round in 0..10_000 {
                        space
                            .write_value(a, if round % 3 == 0 { 8 } else { 1 << lane }, value)
                            .unwrap()
                            .into_ready()
                            .unwrap();
                    }
                });
            }
            for _ in 0..10_000 {
                let value = space.read_value(b, 8).unwrap().into_ready().unwrap();
                assert!(value.to_le_bytes()[..8].iter().all(|&b| b == 0 || b == 255));
            }
        });
    }
    #[test]
    fn native_context_retains_its_address_space_and_recovery_keeps_old_physical_inputs() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(PAGE_SIZE).unwrap();
        let old_id = first.identity();
        let replacement = account.allocate(PAGE_SIZE).unwrap();
        let new_id = replacement.identity();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let at = GuestVirtualAddress::new(0x4000);
        space
            .map(at, first, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
            .unwrap();
        let (interrupt, halted) = std::sync::mpsc::channel();
        let context = space.register_core(Arc::new(move || {
            interrupt.send(()).unwrap();
        }));
        let native = context.enter_native().unwrap();
        let capture = space
            .capture_native(&native, at, 8, MemoryPermissions::READ)
            .unwrap();
        let remapping = Arc::clone(&space);
        let change = std::thread::spawn(move || {
            remapping
                .remap(at, replacement, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
                .unwrap()
        });
        halted
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        native.suspend();
        change.join().unwrap();
        native.resume().unwrap();
        assert_eq!(
            space
                .capture_native(&native, at, 8, MemoryPermissions::READ)
                .unwrap()[0]
                .allocation
                .identity(),
            new_id
        );
        assert_eq!(capture[0].allocation.identity(), old_id);
        drop(space);
        assert_eq!(machine.statistics().guest_owned_bytes, PAGE_SIZE);
        drop(native);
        drop(context);
        assert_eq!(machine.statistics().guest_owned_bytes, 0);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        drop(capture);
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }
    #[test]
    fn aliases_share_physical_identity_and_retained_captures_survive_remap() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let account = machine.account(4 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let a = GuestVirtualAddress::new(0x1000);
        let b = GuestVirtualAddress::new(0x7000);
        space
            .map(
                a,
                allocation.clone(),
                0,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
            .map(
                b,
                allocation.clone(),
                0,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        drop(allocation);
        let before = space.translate(a, MemoryPermissions::READ).unwrap();
        assert_eq!(
            before.physical,
            space
                .translate(b, MemoryPermissions::READ)
                .unwrap()
                .physical
        );
        let capture = space
            .capture(a, 2 * PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        space.unmap(a, 2 * PAGE_SIZE).unwrap();
        let replacement = account.allocate(2 * PAGE_SIZE).unwrap();
        space
            .map(a, replacement, 0, 2 * PAGE_SIZE, MemoryPermissions::READ)
            .unwrap();
        let after = space.translate(a, MemoryPermissions::READ).unwrap();
        assert_ne!(after.allocation, before.allocation);
        assert_ne!(after.generation, before.generation);
        assert_eq!(capture[0].allocation.identity(), before.allocation);
        assert_eq!(
            capture[0].physical_extents().next().unwrap().start,
            before.physical
        );
        assert!(space.translate(a, MemoryPermissions::WRITE).is_err());
        drop(space);
        assert_eq!(machine.statistics().retained_bytes, 2 * PAGE_SIZE);
        drop(capture);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(machine.statistics().virtual_reserved_bytes, 0);
    }
}
