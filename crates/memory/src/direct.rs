//! Linux direct guest-address arenas.
//!
//! The arena is derived host state. Canonical bytes remain owned by
//! [`crate::CanonicalBackingPage`]; mappings here are rebuildable aliases of
//! the canonical `memfd` and never become a second memory authority.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use crate::host_mapped::HostMappedBacking;

/// Guest page granule which the direct backend must represent exactly.
pub const DIRECT_PAGE_SIZE: usize = 4096;
const MIN_DIRECT_VMA_MARGIN: usize = 4096;

/// Construction policy for one process address space.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DirectBackendPolicy {
    /// Construct only checked CPU memory.
    Disabled,
    /// Use direct memory when the host can represent every required contract.
    #[default]
    Preferred,
    /// Fail process construction when direct memory is unavailable.
    Required,
}

/// Immutable CPU memory backend selected for one process address space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuMemoryBackend {
    Checked,
    LinuxDirect,
}

/// Exact data access represented by one host mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectProtection {
    None,
    Read,
    ReadWrite,
}

impl DirectProtection {
    const fn native(self) -> i32 {
        match self {
            Self::None => libc::PROT_NONE,
            Self::Read => libc::PROT_READ,
            Self::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        }
    }
}

/// Host facts required before direct memory can be selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectHostCapabilities {
    host_page_size: usize,
    current_vma_count: usize,
    max_vma_count: usize,
}

impl DirectHostCapabilities {
    /// Detects the host facts used by the direct backend.
    pub fn detect() -> Result<Self, DirectMemoryError> {
        let observed = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let host_page_size = usize::try_from(observed)
            .ok()
            .filter(|size| size.is_power_of_two())
            .ok_or_else(|| DirectMemoryError::invalid("host page size is unavailable"))?;
        let current_vma_count = host_vma_count()?;
        let max_vma_count = std::fs::read_to_string("/proc/sys/vm/max_map_count")
            .map_err(|error| {
                DirectMemoryError::unsupported(format!("Linux VMA limit is unavailable: {error}"))
            })?
            .trim()
            .parse::<usize>()
            .map_err(|error| {
                DirectMemoryError::unsupported(format!("Linux VMA limit is invalid: {error}"))
            })?;
        validate_host_limits(host_page_size, current_vma_count, max_vma_count)?;
        probe_shared_file_views()?;
        Ok(Self {
            host_page_size,
            current_vma_count,
            max_vma_count,
        })
    }

    #[must_use]
    pub const fn host_page_size(self) -> usize {
        self.host_page_size
    }

    #[must_use]
    pub const fn current_vma_count(self) -> usize {
        self.current_vma_count
    }

    #[must_use]
    pub const fn max_vma_count(self) -> usize {
        self.max_vma_count
    }
}

fn validate_host_limits(
    host_page_size: usize,
    current_vma_count: usize,
    max_vma_count: usize,
) -> Result<(), DirectMemoryError> {
    if host_page_size != DIRECT_PAGE_SIZE {
        return Err(DirectMemoryError::unsupported(format!(
            "host page size {host_page_size} cannot represent the {DIRECT_PAGE_SIZE}-byte guest protection granule"
        )));
    }
    if max_vma_count.saturating_sub(current_vma_count) < MIN_DIRECT_VMA_MARGIN {
        return Err(DirectMemoryError::unsupported(format!(
            "Linux VMA margin is below the required {MIN_DIRECT_VMA_MARGIN} entries ({current_vma_count}/{max_vma_count} already used)"
        )));
    }
    Ok(())
}

/// Pointer-only immutable view bound into a CPU engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectAddressSpaceView {
    pub base: usize,
    pub address_space_size: usize,
    /// Flat guest-page index of immutable store-publication controls.
    ///
    /// A zero entry means that direct stores are not eligible for that guest
    /// page. The table contains no permission or visibility state; those are
    /// represented exclusively by the data-arena protection.
    pub store_controls: usize,
}

impl DirectAddressSpaceView {
    /// Computes a host address only after an unsigned confinement proof.
    #[must_use]
    pub fn host_address(self, guest_address: u64) -> Option<usize> {
        let guest = usize::try_from(guest_address).ok()?;
        if guest >= self.address_space_size {
            return None;
        }
        self.base.checked_add(guest)
    }
}

/// Immutable addresses used to publish one completed native CPU store.
///
/// One instance belongs to a canonical physical page and is shared by all of
/// its writable virtual aliases. Generated code obtains its address through
/// [`DirectAddressSpaceView::store_controls`]. None of these fields authorizes
/// an access: semantic access is granted only by the host page protection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectStoreControl {
    pub write_sequence_address: usize,
    pub generation_address: usize,
    pub content_epoch_address: usize,
    pub cpu_write_epoch_address: usize,
    pub cpu_writes_active_address: usize,
    pub write_armed_address: usize,
}

/// One page requested for batched publication.
#[derive(Clone, Copy)]
pub struct DirectMapRequest<'a> {
    pub guest_address: u64,
    pub backing: &'a HostMappedBacking,
    pub protection: DirectProtection,
}

/// One mapped range requested for batched protection publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectProtectRequest {
    pub guest_address: u64,
    pub size: usize,
    pub protection: DirectProtection,
}

struct DirectMappedPage {
    fd: i32,
    backing_offset: u64,
    protection: DirectProtection,
}

struct DirectArenaState {
    pages: BTreeMap<u64, DirectMappedPage>,
}

/// Zero-filled, sparsely committed store-control pointer table.
///
/// Keeping this table flat preserves one indexed load in generated stores. An
/// anonymous `MAP_NORESERVE` mapping avoids allocating or touching metadata for
/// the guest pages which a process never maps.
struct DirectControlTable {
    base: NonNull<AtomicUsize>,
    entry_count: usize,
    byte_size: usize,
}

impl DirectControlTable {
    fn new(entry_count: usize) -> Result<Self, DirectMemoryError> {
        let byte_size = entry_count
            .checked_mul(std::mem::size_of::<AtomicUsize>())
            .ok_or_else(|| DirectMemoryError::invalid("direct control table size overflows"))?;
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(DirectMemoryError::last(
                "direct store-control reservation failed",
            ));
        }
        Ok(Self {
            base: NonNull::new(mapped.cast())
                .expect("mmap never returns null on successful control reservation"),
            entry_count,
            byte_size,
        })
    }

    fn as_ptr(&self) -> *const AtomicUsize {
        self.base.as_ptr()
    }

    fn slot(&self, index: usize) -> &AtomicUsize {
        debug_assert!(index < self.entry_count);
        unsafe { &*self.base.as_ptr().add(index) }
    }
}

unsafe impl Send for DirectControlTable {}
unsafe impl Sync for DirectControlTable {}

impl Drop for DirectControlTable {
    fn drop(&mut self) {
        let result = unsafe { libc::munmap(self.base.as_ptr().cast(), self.byte_size) };
        debug_assert_eq!(result, 0, "direct store-control munmap failed");
    }
}

struct DirectArenaInner {
    reservation: NonNull<u8>,
    reservation_size: usize,
    base: NonNull<u8>,
    address_space_size: usize,
    store_controls: DirectControlTable,
    state: Mutex<DirectArenaState>,
    poisoned: AtomicBool,
    #[cfg(test)]
    fail_after_host_calls: std::sync::atomic::AtomicI64,
}

unsafe impl Send for DirectArenaInner {}
unsafe impl Sync for DirectArenaInner {}

impl Drop for DirectArenaInner {
    fn drop(&mut self) {
        let result =
            unsafe { libc::munmap(self.reservation.as_ptr().cast(), self.reservation_size) };
        debug_assert_eq!(result, 0, "direct arena munmap failed");
    }
}

/// Guarded direct guest-address reservation shared by all physical aliases.
#[derive(Clone)]
pub struct DirectArena {
    inner: Arc<DirectArenaInner>,
}

/// Non-owning arena reference retained by canonical physical pages.
#[derive(Clone)]
pub(crate) struct DirectArenaWeak {
    inner: Weak<DirectArenaInner>,
}

impl DirectArenaWeak {
    pub(crate) fn upgrade(&self) -> Option<DirectArena> {
        Some(DirectArena {
            inner: self.inner.upgrade()?,
        })
    }
}

impl std::fmt::Debug for DirectArena {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DirectArena")
            .field("view", &self.view())
            .field("poisoned", &self.is_poisoned())
            .field("mapped_pages", &self.mapped_pages())
            .finish()
    }
}

impl DirectArena {
    /// Reserves one process-sized arena surrounded by inaccessible guard pages.
    pub fn new(address_space_size: usize) -> Result<Self, DirectMemoryError> {
        DirectHostCapabilities::detect()?;
        if address_space_size == 0 || !address_space_size.is_multiple_of(DIRECT_PAGE_SIZE) {
            return Err(DirectMemoryError::invalid(
                "direct address-space size must be a nonzero guest-page multiple",
            ));
        }
        let reservation_size = address_space_size
            .checked_add(DIRECT_PAGE_SIZE * 2)
            .ok_or_else(|| DirectMemoryError::invalid("direct reservation size overflows"))?;
        let store_controls = DirectControlTable::new(address_space_size / DIRECT_PAGE_SIZE)?;
        let reservation = reserve(reservation_size, "direct arena reservation failed")?;
        let base = unsafe { NonNull::new_unchecked(reservation.as_ptr().add(DIRECT_PAGE_SIZE)) };
        if base
            .as_ptr()
            .addr()
            .checked_add(address_space_size)
            .is_none()
        {
            unsafe { libc::munmap(reservation.as_ptr().cast(), reservation_size) };
            return Err(DirectMemoryError::invalid(
                "direct host address calculation overflows",
            ));
        }
        Ok(Self {
            inner: Arc::new(DirectArenaInner {
                reservation,
                reservation_size,
                base,
                address_space_size,
                store_controls,
                state: Mutex::new(DirectArenaState {
                    pages: BTreeMap::new(),
                }),
                poisoned: AtomicBool::new(false),
                #[cfg(test)]
                fail_after_host_calls: std::sync::atomic::AtomicI64::new(-1),
            }),
        })
    }

    #[must_use]
    pub fn view(&self) -> DirectAddressSpaceView {
        DirectAddressSpaceView {
            base: self.inner.base.as_ptr().addr(),
            address_space_size: self.inner.address_space_size,
            store_controls: self.inner.store_controls.as_ptr().addr(),
        }
    }

    /// Publishes or clears the immutable store control for one mapped page.
    ///
    /// Callers hold the exclusive execution transition while changing the
    /// table, so native code never races a pointer lifetime transition.
    pub(crate) fn publish_store_control(
        &self,
        guest_address: u64,
        control: Option<&DirectStoreControl>,
    ) -> Result<(), DirectMemoryError> {
        self.validate_range(guest_address, DIRECT_PAGE_SIZE)?;
        let index = guest_address as usize / DIRECT_PAGE_SIZE;
        self.inner.store_controls.slot(index).store(
            control.map_or(0, |control| std::ptr::from_ref(control).addr()),
            Ordering::Release,
        );
        Ok(())
    }

    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.inner).addr()
    }

    pub(crate) fn downgrade(&self) -> DirectArenaWeak {
        DirectArenaWeak {
            inner: Arc::downgrade(&self.inner),
        }
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.inner.poisoned.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn mapped_pages(&self) -> usize {
        self.lock_state().pages.len()
    }

    #[must_use]
    pub fn protection_at(&self, guest_address: u64) -> Option<DirectProtection> {
        self.lock_state()
            .pages
            .get(&guest_address)
            .map(|page| page.protection)
    }

    #[must_use]
    pub fn contains_host_address(&self, address: usize) -> bool {
        let view = self.view();
        address
            .checked_sub(view.base)
            .is_some_and(|offset| offset < view.address_space_size)
    }

    #[must_use]
    pub fn guest_address(&self, host_address: usize) -> Option<u64> {
        let offset = host_address.checked_sub(self.view().base)?;
        (offset < self.inner.address_space_size).then_some(offset as u64)
    }

    /// Maps page requests, coalescing adjacent guest and backing offsets.
    pub fn map_pages(&self, requests: &[DirectMapRequest<'_>]) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        if requests.is_empty() {
            return Ok(());
        }
        for request in requests {
            self.validate_range(request.guest_address, DIRECT_PAGE_SIZE)?;
            if request.backing.size() != DIRECT_PAGE_SIZE {
                return Err(DirectMemoryError::invalid(
                    "direct mappings require one canonical guest page",
                ));
            }
        }
        if requests
            .windows(2)
            .any(|pair| pair[0].guest_address >= pair[1].guest_address)
        {
            return Err(DirectMemoryError::invalid(
                "direct map requests must be strictly guest-address ordered",
            ));
        }

        let mut state = self.lock_state();
        let mut start = 0;
        while start < requests.len() {
            let first = &requests[start];
            let mut end = start + 1;
            while end < requests.len() {
                let previous = &requests[end - 1];
                let current = &requests[end];
                let adjacent_guest = previous.guest_address.checked_add(DIRECT_PAGE_SIZE as u64)
                    == Some(current.guest_address);
                let adjacent_backing = previous
                    .backing
                    .offset()
                    .checked_add(DIRECT_PAGE_SIZE as u64)
                    == Some(current.backing.offset());
                if !adjacent_guest
                    || !adjacent_backing
                    || previous.backing.fd() != current.backing.fd()
                    || first.protection != current.protection
                {
                    break;
                }
                end += 1;
            }
            let page_count = end - start;
            let size = page_count
                .checked_mul(DIRECT_PAGE_SIZE)
                .ok_or_else(|| DirectMemoryError::invalid("direct map batch size overflows"))?;
            self.map_run(first, size)?;
            for request in &requests[start..end] {
                state.pages.insert(
                    request.guest_address,
                    DirectMappedPage {
                        fd: request.backing.fd(),
                        backing_offset: request.backing.offset(),
                        protection: request.protection,
                    },
                );
            }
            start = end;
        }
        Ok(())
    }

    /// Reconciles the complete desired page set with the current arena.
    ///
    /// The caller owns the execution transition, so no guest can observe the
    /// intermediate host operations. Any partial host failure poisons the
    /// whole arena before the transition is released.
    pub fn reconcile_pages(
        &self,
        requests: &[DirectMapRequest<'_>],
    ) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        if requests
            .windows(2)
            .any(|pair| pair[0].guest_address >= pair[1].guest_address)
        {
            return Err(DirectMemoryError::invalid(
                "direct reconciliation requests must be strictly guest-address ordered",
            ));
        }

        let (removed, remapped, protected) = {
            let state = self.lock_state();
            let desired = requests
                .iter()
                .map(|request| (request.guest_address, request))
                .collect::<BTreeMap<_, _>>();
            let removed = state
                .pages
                .keys()
                .filter(|guest| !desired.contains_key(guest))
                .copied()
                .collect::<Vec<_>>();
            let mut remapped = Vec::new();
            let mut protected = Vec::new();
            for request in requests {
                match state.pages.get(&request.guest_address) {
                    Some(current)
                        if current.fd == request.backing.fd()
                            && current.backing_offset == request.backing.offset() =>
                    {
                        if current.protection != request.protection {
                            protected.push(DirectProtectRequest {
                                guest_address: request.guest_address,
                                size: DIRECT_PAGE_SIZE,
                                protection: request.protection,
                            });
                        }
                    }
                    _ => remapped.push(*request),
                }
            }
            (removed, remapped, protected)
        };

        let removed = coalesce_pages(&removed, DirectProtection::None);
        self.replace_with_none(&removed)?;
        self.map_pages(&remapped)?;
        self.protect_ranges(&protected)
    }

    /// Reconciles only the listed mappings and leaves every unlisted arena
    /// page unchanged.
    ///
    /// This is used by canonical range mutations which already know their
    /// exact affected pages. Protection-only changes are published as
    /// coalesced `mprotect` runs instead of remapping or protecting each page
    /// separately.
    pub fn reconcile_mapped_pages(
        &self,
        requests: &[DirectMapRequest<'_>],
    ) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        for request in requests {
            self.validate_range(request.guest_address, DIRECT_PAGE_SIZE)?;
            if request.backing.size() != DIRECT_PAGE_SIZE {
                return Err(DirectMemoryError::invalid(
                    "direct mappings require one canonical guest page",
                ));
            }
        }
        if requests
            .windows(2)
            .any(|pair| pair[0].guest_address >= pair[1].guest_address)
        {
            return Err(DirectMemoryError::invalid(
                "partial direct reconciliation requests must be strictly guest-address ordered",
            ));
        }

        let (remapped, protected) = {
            let state = self.lock_state();
            let mut remapped = Vec::new();
            let mut protected = Vec::new();
            for request in requests {
                match state.pages.get(&request.guest_address) {
                    Some(current)
                        if current.fd == request.backing.fd()
                            && current.backing_offset == request.backing.offset() =>
                    {
                        if current.protection != request.protection {
                            protected.push(DirectProtectRequest {
                                guest_address: request.guest_address,
                                size: DIRECT_PAGE_SIZE,
                                protection: request.protection,
                            });
                        }
                    }
                    _ => remapped.push(*request),
                }
            }
            (remapped, protected)
        };

        self.map_pages(&remapped)?;
        self.protect_ranges(&protected)
    }

    /// Reconciles one guest page without inspecting any other mapping.
    pub fn reconcile_page(
        &self,
        guest_address: u64,
        desired: Option<DirectMapRequest<'_>>,
    ) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        self.validate_range(guest_address, DIRECT_PAGE_SIZE)?;
        let current = self
            .lock_state()
            .pages
            .get(&guest_address)
            .map(|page| (page.fd, page.backing_offset, page.protection));
        match (current, desired) {
            (None, None) => Ok(()),
            (Some(_), None) => self.replace_with_none(&[DirectProtectRequest {
                guest_address,
                size: DIRECT_PAGE_SIZE,
                protection: DirectProtection::None,
            }]),
            (_, Some(request)) if request.guest_address != guest_address => Err(
                DirectMemoryError::invalid("direct page request address does not match its key"),
            ),
            (Some((fd, offset, protection)), Some(request))
                if fd == request.backing.fd() && offset == request.backing.offset() =>
            {
                if protection == request.protection {
                    Ok(())
                } else {
                    self.protect_ranges(&[DirectProtectRequest {
                        guest_address,
                        size: DIRECT_PAGE_SIZE,
                        protection: request.protection,
                    }])
                }
            }
            (_, Some(request)) => self.map_pages(&[request]),
        }
    }

    /// Applies exact protection to mapped ranges and coalesces adjacent runs.
    pub fn protect_ranges(
        &self,
        requests: &[DirectProtectRequest],
    ) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        if requests.is_empty() {
            return Ok(());
        }
        for request in requests {
            self.validate_range(request.guest_address, request.size)?;
        }
        if requests.windows(2).any(|pair| {
            pair[0]
                .guest_address
                .checked_add(pair[0].size as u64)
                .is_none_or(|end| end > pair[1].guest_address)
        }) {
            return Err(DirectMemoryError::invalid(
                "direct protection requests must be ordered and non-overlapping",
            ));
        }
        let mut state = self.lock_state();
        let requests = requests
            .iter()
            .copied()
            .filter(|request| {
                pages(request.guest_address, request.size).any(|page| {
                    state
                        .pages
                        .get(&page)
                        .is_some_and(|mapped| mapped.protection != request.protection)
                })
            })
            .collect::<Vec<_>>();
        if requests.is_empty() {
            return Ok(());
        }
        for request in &requests {
            for page in pages(request.guest_address, request.size) {
                if !state.pages.contains_key(&page) {
                    return Err(DirectMemoryError::invalid(
                        "direct protection request contains an unmapped page",
                    ));
                }
            }
        }
        let mut start = 0;
        while start < requests.len() {
            let first = requests[start];
            let mut size = first.size;
            let mut end = start + 1;
            while end < requests.len()
                && requests[end].protection == first.protection
                && first.guest_address.checked_add(size as u64) == Some(requests[end].guest_address)
            {
                size = size.checked_add(requests[end].size).ok_or_else(|| {
                    DirectMemoryError::invalid("direct protection size overflows")
                })?;
                end += 1;
            }
            self.protect_run(first.guest_address, size, first.protection)?;
            for page in pages(first.guest_address, size) {
                let mapped = state
                    .pages
                    .get_mut(&page)
                    .expect("direct protection pages were preflighted");
                mapped.protection = first.protection;
            }
            start = end;
        }
        Ok(())
    }

    /// Replaces mapped ranges with fresh inaccessible anonymous pages.
    pub fn replace_with_none(
        &self,
        requests: &[DirectProtectRequest],
    ) -> Result<(), DirectMemoryError> {
        self.require_healthy()?;
        let mut state = self.lock_state();
        for request in requests {
            self.validate_range(request.guest_address, request.size)?;
        }
        for request in requests {
            self.replace_run_with_none(request.guest_address, request.size)?;
            for page in pages(request.guest_address, request.size) {
                state.pages.remove(&page);
                self.inner
                    .store_controls
                    .slot(page as usize / DIRECT_PAGE_SIZE)
                    .store(0, Ordering::Release);
            }
        }
        Ok(())
    }

    fn map_run(&self, first: &DirectMapRequest<'_>, size: usize) -> Result<(), DirectMemoryError> {
        self.before_host_call()?;
        let destination = self.host_pointer(first.guest_address)?;
        let offset = libc::off_t::try_from(first.backing.offset())
            .map_err(|_| DirectMemoryError::invalid("direct backing offset exceeds off_t"))?;
        let mapped = unsafe {
            libc::mmap(
                destination.cast(),
                size,
                first.protection.native(),
                libc::MAP_SHARED | libc::MAP_FIXED,
                first.backing.fd(),
                offset,
            )
        };
        if mapped == libc::MAP_FAILED || mapped != destination.cast() {
            return self.host_failure("direct range mmap failed");
        }
        Ok(())
    }

    fn protect_run(
        &self,
        guest_address: u64,
        size: usize,
        protection: DirectProtection,
    ) -> Result<(), DirectMemoryError> {
        self.before_host_call()?;
        let destination = self.host_pointer(guest_address)?;
        if unsafe { libc::mprotect(destination.cast(), size, protection.native()) } != 0 {
            return self.host_failure("direct range mprotect failed");
        }
        Ok(())
    }

    fn replace_run_with_none(
        &self,
        guest_address: u64,
        size: usize,
    ) -> Result<(), DirectMemoryError> {
        self.before_host_call()?;
        let destination = self.host_pointer(guest_address)?;
        let mapped = unsafe {
            libc::mmap(
                destination.cast(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED || mapped != destination.cast() {
            return self.host_failure("direct range replacement failed");
        }
        Ok(())
    }

    fn validate_range(&self, guest_address: u64, size: usize) -> Result<(), DirectMemoryError> {
        let start = usize::try_from(guest_address)
            .map_err(|_| DirectMemoryError::invalid("direct guest address exceeds usize"))?;
        if size == 0
            || !start.is_multiple_of(DIRECT_PAGE_SIZE)
            || !size.is_multiple_of(DIRECT_PAGE_SIZE)
            || start
                .checked_add(size)
                .is_none_or(|end| end > self.inner.address_space_size)
        {
            return Err(DirectMemoryError::invalid(
                "direct range is empty, unaligned, overflowing, or outside the reservation",
            ));
        }
        Ok(())
    }

    fn host_pointer(&self, guest_address: u64) -> Result<*mut u8, DirectMemoryError> {
        let guest = usize::try_from(guest_address)
            .map_err(|_| DirectMemoryError::invalid("direct guest address exceeds usize"))?;
        if guest >= self.inner.address_space_size {
            return Err(DirectMemoryError::invalid(
                "direct guest address is outside the reservation",
            ));
        }
        Ok(unsafe { self.inner.base.as_ptr().add(guest) })
    }

    fn require_healthy(&self) -> Result<(), DirectMemoryError> {
        if self.is_poisoned() {
            Err(DirectMemoryError::poisoned())
        } else {
            Ok(())
        }
    }

    fn host_failure<T>(&self, operation: &str) -> Result<T, DirectMemoryError> {
        self.poison();
        Err(DirectMemoryError::last(operation))
    }

    fn poison(&self) {
        if self.inner.poisoned.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = unsafe {
            libc::mprotect(
                self.inner.base.as_ptr().cast(),
                self.inner.address_space_size,
                libc::PROT_NONE,
            )
        };
    }

    fn lock_state(&self) -> MutexGuard<'_, DirectArenaState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(not(test))]
    fn before_host_call(&self) -> Result<(), DirectMemoryError> {
        Ok(())
    }

    #[cfg(test)]
    fn before_host_call(&self) -> Result<(), DirectMemoryError> {
        let remaining = self.inner.fail_after_host_calls.load(Ordering::Relaxed);
        if remaining < 0 {
            return Ok(());
        }
        if remaining == 0 {
            return self.host_failure("injected direct host operation failed");
        }
        self.inner
            .fail_after_host_calls
            .fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(test)]
    fn fail_after_host_calls(&self, successful_calls: i64) {
        self.inner
            .fail_after_host_calls
            .store(successful_calls, Ordering::Relaxed);
    }
}

/// Construction or publication failure of the direct backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMemoryError {
    kind: DirectMemoryErrorKind,
    detail: Box<str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectMemoryErrorKind {
    Unsupported,
    Invalid,
    Host,
    Poisoned,
}

impl DirectMemoryError {
    fn unsupported(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectMemoryErrorKind::Unsupported,
            detail: detail.into(),
        }
    }

    fn invalid(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectMemoryErrorKind::Invalid,
            detail: detail.into(),
        }
    }

    pub(crate) fn invalid_contract(detail: impl Into<Box<str>>) -> Self {
        Self::invalid(detail)
    }

    fn last(operation: &str) -> Self {
        Self {
            kind: DirectMemoryErrorKind::Host,
            detail: format!("{operation}: {}", std::io::Error::last_os_error()).into_boxed_str(),
        }
    }

    fn poisoned() -> Self {
        Self {
            kind: DirectMemoryErrorKind::Poisoned,
            detail: "direct address space is poisoned after an uncertain host operation".into(),
        }
    }

    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(self.kind, DirectMemoryErrorKind::Unsupported)
    }
}

impl Display for DirectMemoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for DirectMemoryError {}

fn host_vma_count() -> Result<usize, DirectMemoryError> {
    std::fs::read_to_string("/proc/self/maps")
        .map(|maps| maps.lines().count())
        .map_err(|error| {
            DirectMemoryError::unsupported(format!("Linux VMA inventory is unavailable: {error}"))
        })
}

fn probe_shared_file_views() -> Result<(), DirectMemoryError> {
    static PROBE: std::sync::OnceLock<Result<(), DirectMemoryError>> = std::sync::OnceLock::new();
    PROBE.get_or_init(probe_shared_file_views_once).clone()
}

fn probe_shared_file_views_once() -> Result<(), DirectMemoryError> {
    let name = CString::new("nixe-direct-capability")
        .expect("direct capability probe name contains no NUL");
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(DirectMemoryError::unsupported(format!(
            "shared memfd views are unavailable: {}",
            std::io::Error::last_os_error()
        )));
    }
    if unsafe { libc::ftruncate(fd, DIRECT_PAGE_SIZE as libc::off_t) } != 0 {
        let error = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(DirectMemoryError::unsupported(format!(
            "shared memfd sizing is unavailable: {error}"
        )));
    }
    let first = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            DIRECT_PAGE_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    let second = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            DIRECT_PAGE_SIZE,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    let valid = first != libc::MAP_FAILED && second != libc::MAP_FAILED;
    let coherent = if valid {
        unsafe {
            first.cast::<u8>().write_volatile(0x5a);
            second.cast::<u8>().read_volatile() == 0x5a
        }
    } else {
        false
    };
    if first != libc::MAP_FAILED {
        unsafe { libc::munmap(first, DIRECT_PAGE_SIZE) };
    }
    if second != libc::MAP_FAILED {
        unsafe { libc::munmap(second, DIRECT_PAGE_SIZE) };
    }
    unsafe { libc::close(fd) };
    if !valid || !coherent {
        return Err(DirectMemoryError::unsupported(format!(
            "coherent shared-file aliases are unavailable: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

fn reserve(size: usize, operation: &str) -> Result<NonNull<u8>, DirectMemoryError> {
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(DirectMemoryError::last(operation));
    }
    Ok(NonNull::new(mapped.cast()).expect("mmap never returns null on success"))
}

fn pages(start: u64, size: usize) -> impl Iterator<Item = u64> {
    let count = size / DIRECT_PAGE_SIZE;
    (0..count).map(move |page| start + (page * DIRECT_PAGE_SIZE) as u64)
}

fn coalesce_pages(pages: &[u64], protection: DirectProtection) -> Vec<DirectProtectRequest> {
    let mut ranges = Vec::new();
    for &page in pages {
        match ranges.last_mut() {
            Some(DirectProtectRequest {
                guest_address,
                size,
                ..
            }) if guest_address.checked_add(*size as u64) == Some(page) => {
                *size += DIRECT_PAGE_SIZE;
            }
            _ => ranges.push(DirectProtectRequest {
                guest_address: page,
                size: DIRECT_PAGE_SIZE,
                protection,
            }),
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalBackingPage, CanonicalBackingStore, ContentGeneration, GuestPhysicalPageId,
    };

    fn page(store: &CanonicalBackingStore, id: u64, byte: u8) -> CanonicalBackingPage {
        CanonicalBackingPage::initialized(
            store,
            GuestPhysicalPageId::new(id),
            &vec![byte; DIRECT_PAGE_SIZE],
            ContentGeneration::new(1),
        )
        .unwrap()
    }

    #[test]
    fn guarded_process_sized_arena_confines_host_addresses() {
        let arena = DirectArena::new(0x20_000).unwrap();
        let view = arena.view();
        assert_eq!(view.address_space_size, 0x20_000);
        assert_eq!(view.host_address(0), Some(view.base));
        assert_eq!(view.host_address(0x1_ffff), Some(view.base + 0x1_ffff));
        assert_eq!(view.host_address(0x20_000), None);
        assert!(!arena.contains_host_address(view.base - 1));
        assert!(arena.contains_host_address(view.base));
        assert!(!arena.contains_host_address(view.base + 0x20_000));
    }

    #[test]
    fn thirty_nine_bit_control_table_is_sparse_and_keeps_one_index_lookup() {
        const ADDRESS_SPACE_SIZE: usize = 1_usize << 39;
        let arena = DirectArena::new(ADDRESS_SPACE_SIZE).unwrap();
        let view = arena.view();
        let first = unsafe { &*(view.store_controls as *const AtomicUsize) };
        let last = unsafe {
            &*(view.store_controls as *const AtomicUsize)
                .add(ADDRESS_SPACE_SIZE / DIRECT_PAGE_SIZE - 1)
        };

        assert_eq!(first.load(Ordering::Relaxed), 0);
        assert_eq!(last.load(Ordering::Relaxed), 0);
        assert_eq!(arena.mapped_pages(), 0);
    }

    #[test]
    fn host_capability_validation_rejects_inexact_pages_and_low_vma_margin() {
        assert!(validate_host_limits(DIRECT_PAGE_SIZE, 100, 100 + MIN_DIRECT_VMA_MARGIN).is_ok());
        assert!(validate_host_limits(DIRECT_PAGE_SIZE * 4, 100, 100_000).is_err());
        assert!(
            validate_host_limits(DIRECT_PAGE_SIZE, 100, 100 + MIN_DIRECT_VMA_MARGIN - 1).is_err()
        );
    }

    #[test]
    fn adjacent_backing_pages_are_mapped() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let first = page(&store, 1, 0x11);
        let second = page(&store, 2, 0x22);
        let first = first.direct_backing().unwrap();
        let second = second.direct_backing().unwrap();
        let arena = DirectArena::new(0x20_000).unwrap();
        arena
            .map_pages(&[
                DirectMapRequest {
                    guest_address: 0x4000,
                    backing: &first,
                    protection: DirectProtection::Read,
                },
                DirectMapRequest {
                    guest_address: 0x5000,
                    backing: &second,
                    protection: DirectProtection::Read,
                },
            ])
            .unwrap();
        let view = arena.view();
        assert_eq!(unsafe { *((view.base + 0x4000) as *const u8) }, 0x11);
        assert_eq!(unsafe { *((view.base + 0x5000) as *const u8) }, 0x22);
    }

    #[test]
    fn aliases_share_bytes_and_protection() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = page(&store, 1, 0x33);
        let backing = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x20_000).unwrap();
        arena
            .map_pages(&[
                DirectMapRequest {
                    guest_address: 0x4000,
                    backing: &backing,
                    protection: DirectProtection::ReadWrite,
                },
                DirectMapRequest {
                    guest_address: 0x8000,
                    backing: &backing,
                    protection: DirectProtection::ReadWrite,
                },
            ])
            .unwrap();
        arena
            .protect_ranges(&[
                DirectProtectRequest {
                    guest_address: 0x4000,
                    size: DIRECT_PAGE_SIZE,
                    protection: DirectProtection::ReadWrite,
                },
                DirectProtectRequest {
                    guest_address: 0x8000,
                    size: DIRECT_PAGE_SIZE,
                    protection: DirectProtection::ReadWrite,
                },
            ])
            .unwrap();
        let view = arena.view();
        unsafe { *((view.base + 0x4000) as *mut u8) = 0x77 };
        assert_eq!(unsafe { *((view.base + 0x8000) as *const u8) }, 0x77);
        arena
            .protect_ranges(&[
                DirectProtectRequest {
                    guest_address: 0x4000,
                    size: DIRECT_PAGE_SIZE,
                    protection: DirectProtection::Read,
                },
                DirectProtectRequest {
                    guest_address: 0x8000,
                    size: DIRECT_PAGE_SIZE,
                    protection: DirectProtection::Read,
                },
            ])
            .unwrap();
        assert_eq!(arena.protection_at(0x4000), Some(DirectProtection::Read));
        assert_eq!(arena.protection_at(0x8000), Some(DirectProtection::Read));
    }

    #[test]
    fn failed_partial_publication_poisons_the_complete_arena() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let first = page(&store, 1, 0x11);
        let second = page(&store, 2, 0x22);
        let first = first.direct_backing().unwrap();
        let second = second.direct_backing().unwrap();
        let arena = DirectArena::new(0x20_000).unwrap();
        arena.fail_after_host_calls(1);
        let error = arena
            .map_pages(&[
                DirectMapRequest {
                    guest_address: 0x4000,
                    backing: &first,
                    protection: DirectProtection::Read,
                },
                DirectMapRequest {
                    guest_address: 0x8000,
                    backing: &second,
                    protection: DirectProtection::Read,
                },
            ])
            .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert!(arena.is_poisoned());
    }
}
