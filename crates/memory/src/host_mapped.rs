//! Linux shared-file storage for canonical guest bytes.

use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::DIRECT_PAGE_SIZE;

const HOST_BACKING_RESERVATION_SIZE: usize = 1usize << 39;
const HOST_BACKING_GROWTH_SIZE: usize = 1 << 26;

#[cfg(target_arch = "x86_64")]
pub(crate) fn host_atomic_128_supported() -> bool {
    std::arch::is_x86_feature_detected!("cmpxchg16b")
}

#[cfg(target_arch = "aarch64")]
pub(crate) const fn host_atomic_128_supported() -> bool {
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) const fn host_atomic_128_supported() -> bool {
    false
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

    pub(crate) fn atomic_load(&self, offset: usize, size: usize) -> Result<u128, HostMappedError> {
        let pointer = self.atomic_pointer(offset, size)?;
        let value = unsafe {
            match size {
                1 => u128::from(AtomicU8::from_ptr(pointer).load(Ordering::Acquire)),
                2 => u128::from(AtomicU16::from_ptr(pointer.cast()).load(Ordering::Acquire)),
                4 => u128::from(AtomicU32::from_ptr(pointer.cast()).load(Ordering::Acquire)),
                8 => u128::from(AtomicU64::from_ptr(pointer.cast()).load(Ordering::Acquire)),
                16 => atomic_load_128(pointer.cast()),
                _ => unreachable!("atomic_pointer validates the width"),
            }
        };
        Ok(value)
    }

    pub(crate) fn atomic_compare_exchange(
        &self,
        offset: usize,
        size: usize,
        expected: u128,
        replacement: u128,
    ) -> Result<(u128, bool), HostMappedError> {
        let pointer = self.atomic_pointer(offset, size)?;
        let (observed, stored) = unsafe {
            match size {
                1 => match AtomicU8::from_ptr(pointer).compare_exchange(
                    expected as u8,
                    replacement as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(value) => (u128::from(value), true),
                    Err(value) => (u128::from(value), false),
                },
                2 => match AtomicU16::from_ptr(pointer.cast()).compare_exchange(
                    expected as u16,
                    replacement as u16,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(value) => (u128::from(value), true),
                    Err(value) => (u128::from(value), false),
                },
                4 => match AtomicU32::from_ptr(pointer.cast()).compare_exchange(
                    expected as u32,
                    replacement as u32,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(value) => (u128::from(value), true),
                    Err(value) => (u128::from(value), false),
                },
                8 => match AtomicU64::from_ptr(pointer.cast()).compare_exchange(
                    expected as u64,
                    replacement as u64,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(value) => (u128::from(value), true),
                    Err(value) => (u128::from(value), false),
                },
                16 => atomic_compare_exchange_128(pointer.cast(), expected, replacement),
                _ => unreachable!("atomic_pointer validates the width"),
            }
        };
        Ok((observed, stored))
    }

    fn atomic_pointer(&self, offset: usize, size: usize) -> Result<*mut u8, HostMappedError> {
        if !matches!(size, 1 | 2 | 4 | 8 | 16) {
            return Err(HostMappedError::invalid("unsupported host atomic width"));
        }
        if size == 16 && !host_atomic_128_supported() {
            return Err(HostMappedError::invalid(
                "128-bit host atomics require CMPXCHG16B on x86-64",
            ));
        }
        offset
            .checked_add(size)
            .filter(|end| *end <= self.size())
            .ok_or_else(|| HostMappedError::invalid("host atomic range is out of bounds"))?;
        let pointer = unsafe { self.inner.base.as_ptr().add(offset) };
        if pointer.addr() & (size - 1) != 0 {
            return Err(HostMappedError::invalid(
                "host atomic address is not naturally aligned",
            ));
        }
        Ok(pointer)
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn atomic_load_128(pointer: *mut u128) -> u128 {
    unsafe { atomic_compare_exchange_128(pointer, 0, 0).0 }
}

#[cfg(target_arch = "x86_64")]
unsafe fn atomic_compare_exchange_128(
    pointer: *mut u128,
    expected: u128,
    replacement: u128,
) -> (u128, bool) {
    let mut observed = 0_u128;
    let stored = unsafe {
        nixe_memory_compare_exchange_128(
            pointer,
            expected as u64,
            (expected >> 64) as u64,
            replacement as u64,
            (replacement >> 64) as u64,
            &mut observed,
        )
    };
    (observed, stored != 0)
}

#[cfg(target_arch = "aarch64")]
unsafe fn atomic_load_128(pointer: *mut u128) -> u128 {
    unsafe { atomic_compare_exchange_128(pointer, 0, 0).0 }
}

#[cfg(target_arch = "aarch64")]
unsafe fn atomic_compare_exchange_128(
    pointer: *mut u128,
    expected: u128,
    replacement: u128,
) -> (u128, bool) {
    let mut observed = 0_u128;
    let stored = unsafe {
        nixe_memory_compare_exchange_128(
            pointer,
            expected as u64,
            (expected >> 64) as u64,
            replacement as u64,
            (replacement >> 64) as u64,
            &mut observed,
        )
    };
    (observed, stored != 0)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn atomic_load_128(_pointer: *mut u128) -> u128 {
    unreachable!("128-bit host atomics are unsupported on this host ISA")
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn atomic_compare_exchange_128(
    _pointer: *mut u128,
    _expected: u128,
    _replacement: u128,
) -> (u128, bool) {
    unreachable!("128-bit host atomics are unsupported on this host ISA")
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
unsafe extern "C" {
    fn nixe_memory_compare_exchange_128(
        pointer: *mut u128,
        expected_low: u64,
        expected_high: u64,
        replacement_low: u64,
        replacement_high: u64,
        observed: *mut u128,
    ) -> u32;
}

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_memory_compare_exchange_128
    .type nixe_memory_compare_exchange_128,@function
nixe_memory_compare_exchange_128:
    push rbx
    mov rax,rsi
    mov rbx,rcx
    mov rcx,r8
    lock cmpxchg16b [rdi]
    setz r10b
    mov [r9],rax
    mov [r9+8],rdx
    movzx eax,r10b
    pop rbx
    ret
    .size nixe_memory_compare_exchange_128,.-nixe_memory_compare_exchange_128
"#
);

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_memory_compare_exchange_128
    .type nixe_memory_compare_exchange_128,%function
nixe_memory_compare_exchange_128:
1:
    ldaxp x6,x7,[x0]
    cmp x6,x1
    ccmp x7,x2,#0,eq
    b.ne 2f
    stlxp w8,x3,x4,[x0]
    cbnz w8,1b
    mov w0,#1
    b 3f
2:
    clrex
    mov w0,#0
3:
    stp x6,x7,[x5]
    ret
    .size nixe_memory_compare_exchange_128,.-nixe_memory_compare_exchange_128
"#
);

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
    use super::HostMappedStore;

    #[test]
    fn canonical_host_atomics_cover_every_architectural_width() {
        let store = HostMappedStore::new().unwrap();
        let backing = store.allocate(4096, None).unwrap();
        for (offset, size, value) in [
            (0, 1, 0xa5),
            (2, 2, 0xa5c3),
            (4, 4, 0xa5c3_9678),
            (8, 8, 0xa5c3_9678_1234_fedc),
            (16, 16, 0xa5c3_9678_1234_fedc_5a3c_6987_edcb_0123),
        ] {
            assert_eq!(backing.atomic_load(offset, size).unwrap(), 0);
            assert_eq!(
                backing
                    .atomic_compare_exchange(offset, size, 0, value)
                    .unwrap(),
                (0, true)
            );
            assert_eq!(backing.atomic_load(offset, size).unwrap(), value);
            assert_eq!(
                backing
                    .atomic_compare_exchange(offset, size, 0, !value)
                    .unwrap(),
                (value, false)
            );
        }
    }
}
