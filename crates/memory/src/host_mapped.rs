//! Linux host-mapped storage and guest virtual-address arenas.

use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::MemoryPermissions;
use crate::backing::FastmemPageLease;

pub const FASTMEM_ADDRESS_SPACE_BITS: u32 = 39;
pub const FASTMEM_SIZE: usize = 1usize << FASTMEM_ADDRESS_SPACE_BITS;
pub const FASTMEM_PAGE_BITS: u32 = 12;
pub const FASTMEM_PAGE_SIZE: usize = 1usize << FASTMEM_PAGE_BITS;
const FASTMEM_PAGE_COUNT: usize = FASTMEM_SIZE >> FASTMEM_PAGE_BITS;
const HOST_BACKING_RESERVATION_SIZE: usize = FASTMEM_SIZE;
const HOST_BACKING_GROWTH_SIZE: usize = 1 << 26;

pub const FASTMEM_READ: u32 = 1 << 0;
pub const FASTMEM_WRITE: u32 = 1 << 1;

#[repr(C)]
pub struct FastmemEntry {
    pub validity_address: AtomicUsize,
    pub visibility_epoch: AtomicU64,
    pub generation_address: AtomicUsize,
    pub content_epoch_address: AtomicUsize,
    pub cpu_write_epoch_address: AtomicUsize,
    pub cpu_writes_active_address: AtomicUsize,
    pub write_sequence_address: AtomicUsize,
    pub flags: AtomicU32,
    pub reserved: AtomicU32,
}

impl FastmemEntry {
    fn empty() -> Self {
        Self {
            validity_address: AtomicUsize::new(0),
            visibility_epoch: AtomicU64::new(0),
            generation_address: AtomicUsize::new(0),
            content_epoch_address: AtomicUsize::new(0),
            cpu_write_epoch_address: AtomicUsize::new(0),
            cpu_writes_active_address: AtomicUsize::new(0),
            write_sequence_address: AtomicUsize::new(0),
            flags: AtomicU32::new(0),
            reserved: AtomicU32::new(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct FastmemView {
    pub base: usize,
    pub entries: usize,
    pub address_space_size: usize,
}

#[derive(Debug)]
pub struct HostMappedError(Box<str>);

impl HostMappedError {
    fn last(operation: &str) -> Self {
        Self(format!("{operation}: {}", std::io::Error::last_os_error()).into_boxed_str())
    }

    fn invalid(detail: &str) -> Self {
        Self(detail.into())
    }
}

impl Display for HostMappedError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for HostMappedError {}

#[derive(Debug)]
struct HostMappedStoreInner {
    fd: OwnedFd,
    canonical_base: NonNull<u8>,
    allocation: Mutex<HostMappedAllocation>,
}

unsafe impl Send for HostMappedStoreInner {}
unsafe impl Sync for HostMappedStoreInner {}

#[derive(Debug, Default)]
struct HostMappedAllocation {
    next_offset: u64,
    mapped_capacity: usize,
}

impl Drop for HostMappedStoreInner {
    fn drop(&mut self) {
        let result = unsafe {
            libc::munmap(
                self.canonical_base.as_ptr().cast(),
                HOST_BACKING_RESERVATION_SIZE,
            )
        };
        debug_assert_eq!(result, 0, "canonical arena munmap failed");
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HostMappedStore {
    inner: Arc<HostMappedStoreInner>,
}

impl HostMappedStore {
    pub(crate) fn new() -> Result<Self, HostMappedError> {
        let name = CString::new("nixe-canonical-memory").expect("static memfd name has no NUL");
        let raw = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        if raw < 0 {
            return Err(HostMappedError::last("memfd_create failed"));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let canonical_base = reserve(
            HOST_BACKING_RESERVATION_SIZE,
            "canonical backing arena mmap failed",
        )?;
        Ok(Self {
            inner: Arc::new(HostMappedStoreInner {
                fd,
                canonical_base,
                allocation: Mutex::new(HostMappedAllocation::default()),
            }),
        })
    }

    pub(crate) fn allocate(
        &self,
        size: usize,
        contents: Option<&[u8]>,
    ) -> Result<HostMappedBacking, HostMappedError> {
        if size == 0 {
            return Err(HostMappedError::invalid(
                "host-mapped backing size must be nonzero",
            ));
        }
        if contents.is_some_and(|contents| contents.len() != size) {
            return Err(HostMappedError::invalid(
                "host-mapped initial contents must fill the backing",
            ));
        }
        let mut allocation = self
            .inner
            .allocation
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mapped_size = size
            .checked_add(FASTMEM_PAGE_SIZE - 1)
            .map(|value| value & !(FASTMEM_PAGE_SIZE - 1))
            .ok_or_else(|| HostMappedError::invalid("host-mapped backing size overflow"))?;
        let offset = allocation.next_offset;
        let end = offset
            .checked_add(mapped_size as u64)
            .ok_or_else(|| HostMappedError::invalid("host-mapped backing offset overflow"))?;
        let end_usize = usize::try_from(end)
            .map_err(|_| HostMappedError::invalid("host-mapped backing exceeds usize"))?;
        if end_usize > HOST_BACKING_RESERVATION_SIZE {
            return Err(HostMappedError::invalid(
                "canonical backing arena capacity exceeded",
            ));
        }
        if end_usize > allocation.mapped_capacity {
            let new_capacity = end_usize
                .checked_add(HOST_BACKING_GROWTH_SIZE - 1)
                .map(|value| value & !(HOST_BACKING_GROWTH_SIZE - 1))
                .ok_or_else(|| HostMappedError::invalid("canonical capacity overflow"))?
                .min(HOST_BACKING_RESERVATION_SIZE);
            let capacity_off = libc::off_t::try_from(new_capacity)
                .map_err(|_| HostMappedError::invalid("canonical capacity exceeds off_t"))?;
            if unsafe { libc::ftruncate(self.inner.fd.as_raw_fd(), capacity_off) } != 0 {
                return Err(HostMappedError::last("ftruncate failed"));
            }
            let extension = new_capacity - allocation.mapped_capacity;
            let destination = unsafe {
                self.inner
                    .canonical_base
                    .as_ptr()
                    .add(allocation.mapped_capacity)
            };
            let file_offset = libc::off_t::try_from(allocation.mapped_capacity)
                .map_err(|_| HostMappedError::invalid("canonical offset exceeds off_t"))?;
            let mapped = unsafe {
                libc::mmap(
                    destination.cast(),
                    extension,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    self.inner.fd.as_raw_fd(),
                    file_offset,
                )
            };
            if mapped == libc::MAP_FAILED || mapped != destination.cast() {
                return Err(HostMappedError::last("canonical mmap extension failed"));
            }
            allocation.mapped_capacity = new_capacity;
        }
        let base = unsafe {
            NonNull::new_unchecked(self.inner.canonical_base.as_ptr().add(offset as usize))
        };
        if let Some(contents) = contents {
            unsafe { std::ptr::copy_nonoverlapping(contents.as_ptr(), base.as_ptr(), size) };
        }
        allocation.next_offset = end;
        Ok(HostMappedBacking {
            inner: Arc::new(HostMappedBackingInner {
                store: self.clone(),
                offset,
                base,
                size,
            }),
        })
    }
}

#[derive(Debug)]
struct HostMappedBackingInner {
    store: HostMappedStore,
    offset: u64,
    base: NonNull<u8>,
    size: usize,
}

unsafe impl Send for HostMappedBackingInner {}
unsafe impl Sync for HostMappedBackingInner {}

#[derive(Clone, Debug)]
pub struct HostMappedBacking {
    inner: Arc<HostMappedBackingInner>,
}

impl HostMappedBacking {
    #[must_use]
    pub fn base(&self) -> usize {
        self.inner.base.as_ptr().addr()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size
    }

    pub(crate) fn fd(&self) -> i32 {
        self.inner.store.inner.fd.as_raw_fd()
    }

    pub(crate) fn offset(&self) -> u64 {
        self.inner.offset
    }
}

#[derive(Debug)]
pub struct FastmemArena {
    base: NonNull<u8>,
    entries: NonNull<FastmemEntry>,
    mapped_offsets: HashMap<u64, u64>,
}

unsafe impl Send for FastmemArena {}
unsafe impl Sync for FastmemArena {}

impl FastmemArena {
    pub fn new() -> Result<Self, HostMappedError> {
        let base = reserve(FASTMEM_SIZE, "fastmem arena mmap failed")?;
        let entries_size = FASTMEM_PAGE_COUNT
            .checked_mul(size_of::<FastmemEntry>())
            .ok_or_else(|| HostMappedError::invalid("fastmem metadata size overflow"))?;
        let entries = reserve(entries_size, "fastmem metadata mmap failed")?;
        if unsafe {
            libc::mprotect(
                entries.as_ptr().cast(),
                entries_size,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        } != 0
        {
            unsafe { libc::munmap(base.as_ptr().cast(), FASTMEM_SIZE) };
            unsafe { libc::munmap(entries.as_ptr().cast(), entries_size) };
            return Err(HostMappedError::last("fastmem metadata mprotect failed"));
        }
        Ok(Self {
            base,
            entries: entries.cast(),
            mapped_offsets: HashMap::new(),
        })
    }

    #[must_use]
    pub fn view(&self) -> FastmemView {
        FastmemView {
            base: self.base.as_ptr().addr(),
            entries: self.entries.as_ptr().addr(),
            address_space_size: FASTMEM_SIZE,
        }
    }

    pub fn map_page(
        &mut self,
        virtual_address: u64,
        backing: &HostMappedBacking,
    ) -> Result<(), HostMappedError> {
        self.validate_page(virtual_address)?;
        if backing.size() != FASTMEM_PAGE_SIZE {
            return Err(HostMappedError::invalid(
                "fastmem mappings require one host page of backing",
            ));
        }
        if self.mapped_offsets.get(&virtual_address) == Some(&backing.offset()) {
            return Ok(());
        }
        if self.mapped_offsets.contains_key(&virtual_address) {
            self.clear_entry(virtual_address);
        }
        let destination = unsafe { self.base.as_ptr().add(virtual_address as usize) };
        let offset = libc::off_t::try_from(backing.offset())
            .map_err(|_| HostMappedError::invalid("fastmem backing offset exceeds off_t"))?;
        let mapped = unsafe {
            libc::mmap(
                destination.cast(),
                FASTMEM_PAGE_SIZE,
                libc::PROT_NONE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                backing.fd(),
                offset,
            )
        };
        if mapped == libc::MAP_FAILED || mapped != destination.cast() {
            return Err(HostMappedError::last("fastmem page mmap failed"));
        }
        unsafe { self.entry(virtual_address).write(FastmemEntry::empty()) };
        self.mapped_offsets
            .insert(virtual_address, backing.offset());
        Ok(())
    }

    pub fn unmap_page(&mut self, virtual_address: u64) -> Result<(), HostMappedError> {
        self.validate_page(virtual_address)?;
        if !self.mapped_offsets.contains_key(&virtual_address) {
            return Ok(());
        }
        self.clear_entry(virtual_address);
        let destination = unsafe { self.base.as_ptr().add(virtual_address as usize) };
        let mapped = unsafe {
            libc::mmap(
                destination.cast(),
                FASTMEM_PAGE_SIZE,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED || mapped != destination.cast() {
            return Err(HostMappedError::last(
                "fastmem page unmap replacement failed",
            ));
        }
        self.mapped_offsets.remove(&virtual_address);
        Ok(())
    }

    pub fn arm_page(
        &mut self,
        virtual_address: u64,
        permissions: MemoryPermissions,
        lease: &FastmemPageLease,
        write_armed: bool,
    ) -> Result<(), HostMappedError> {
        self.validate_page(virtual_address)?;
        if !self.mapped_offsets.contains_key(&virtual_address) {
            return Err(HostMappedError::invalid(
                "fastmem page must be mapped before it is armed",
            ));
        }
        let mut flags = 0;
        if permissions.contains(MemoryPermissions::READ) {
            flags |= FASTMEM_READ;
        }
        if write_armed && permissions.contains(MemoryPermissions::WRITE) {
            flags |= FASTMEM_WRITE;
        }
        let mut protection = libc::PROT_NONE;
        if flags & FASTMEM_READ != 0 {
            protection |= libc::PROT_READ;
        }
        if flags & FASTMEM_WRITE != 0 {
            protection |= libc::PROT_WRITE;
        }
        let entry = unsafe { &*self.entry(virtual_address) };
        if entry.flags.load(Ordering::Acquire) != flags {
            let destination = unsafe { self.base.as_ptr().add(virtual_address as usize) };
            if unsafe { libc::mprotect(destination.cast(), FASTMEM_PAGE_SIZE, protection) } != 0 {
                return Err(HostMappedError::last("fastmem page mprotect failed"));
            }
        }
        entry
            .validity_address
            .store(lease.validity_address(), Ordering::Relaxed);
        entry
            .visibility_epoch
            .store(lease.visibility_epoch(), Ordering::Relaxed);
        entry
            .generation_address
            .store(lease.generation_address(), Ordering::Relaxed);
        entry
            .content_epoch_address
            .store(lease.content_epoch_address(), Ordering::Relaxed);
        entry
            .cpu_write_epoch_address
            .store(lease.cpu_write_epoch_address(), Ordering::Relaxed);
        entry
            .cpu_writes_active_address
            .store(lease.cpu_writes_active_address(), Ordering::Relaxed);
        entry
            .write_sequence_address
            .store(lease.write_sequence_address(), Ordering::Relaxed);
        entry.flags.store(flags, Ordering::Release);
        Ok(())
    }

    fn validate_page(&self, virtual_address: u64) -> Result<(), HostMappedError> {
        if virtual_address as usize >= FASTMEM_SIZE
            || !virtual_address.is_multiple_of(FASTMEM_PAGE_SIZE as u64)
        {
            return Err(HostMappedError::invalid(
                "fastmem virtual address is outside the aligned arena",
            ));
        }
        Ok(())
    }

    unsafe fn entry(&self, virtual_address: u64) -> *mut FastmemEntry {
        unsafe {
            self.entries
                .as_ptr()
                .add((virtual_address as usize) >> FASTMEM_PAGE_BITS)
        }
    }

    fn clear_entry(&mut self, virtual_address: u64) {
        unsafe { &*self.entry(virtual_address) }
            .flags
            .store(0, Ordering::Release);
    }
}

impl Drop for FastmemArena {
    fn drop(&mut self) {
        let entries_size = FASTMEM_PAGE_COUNT * size_of::<FastmemEntry>();
        let first = unsafe { libc::munmap(self.base.as_ptr().cast(), FASTMEM_SIZE) };
        let second = unsafe { libc::munmap(self.entries.as_ptr().cast(), entries_size) };
        debug_assert_eq!(first, 0, "fastmem arena munmap failed");
        debug_assert_eq!(second, 0, "fastmem metadata munmap failed");
    }
}

fn reserve(size: usize, operation: &str) -> Result<NonNull<u8>, HostMappedError> {
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
        return Err(HostMappedError::last(operation));
    }
    Ok(NonNull::new(mapped.cast()).expect("mmap never returns null on success"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_share_canonical_bytes_and_permissions_are_armed_explicitly() {
        let store = crate::CanonicalBackingStore::allocate().unwrap();
        let page = crate::CanonicalBackingPage::initialized(
            &store,
            crate::GuestPhysicalPageId::new(1),
            &vec![0x11; FASTMEM_PAGE_SIZE],
            crate::ContentGeneration::new(1),
        )
        .unwrap();
        page.prepare_write().unwrap();
        page.write_preflighted(
            0,
            &[0x11],
            crate::ContentGeneration::new(1),
            crate::ContentGeneration::new(2),
        )
        .unwrap();
        let lease = page
            .acquire_fastmem(MemoryPermissions::READ_WRITE, true)
            .unwrap()
            .unwrap();
        let mut arena = FastmemArena::new().unwrap();
        arena.map_page(0x4000, lease.host_mapped_backing()).unwrap();
        arena
            .arm_page(0x4000, MemoryPermissions::READ_WRITE, &lease, true)
            .unwrap();
        let guest = (arena.view().base + 0x4000) as *mut u8;
        assert_eq!(unsafe { guest.read() }, 0x11);
        unsafe { guest.write(0x77) };
        assert_eq!(
            unsafe { (lease.host_mapped_backing().base() as *const u8).read() },
            0x77
        );
        arena.unmap_page(0x4000).unwrap();
        assert_eq!(
            unsafe { &*arena.entry(0x4000) }
                .flags
                .load(Ordering::Acquire),
            0
        );
    }
}
