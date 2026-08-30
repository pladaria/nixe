//! Linux shared-file storage for canonical guest bytes.

use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};

use crate::DIRECT_PAGE_SIZE;

const HOST_BACKING_RESERVATION_SIZE: usize = 1usize << 39;
const HOST_BACKING_GROWTH_SIZE: usize = 1 << 26;

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
            .checked_add(DIRECT_PAGE_SIZE - 1)
            .map(|value| value & !(DIRECT_PAGE_SIZE - 1))
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
