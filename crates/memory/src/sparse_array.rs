//! Zero-backed, directly indexed page-table storage, as SparseLargeVector and
//! MultiLevelPageTable reserve their native arrays on Linux.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/sparse_large_vector.h
use crate::Machine;
use std::{ops::Index, ptr::NonNull};

#[derive(Debug)]
pub struct SparseArray<T> {
    machine: Machine,
    base: NonNull<T>,
    entries: usize,
    bytes: usize,
}
// SAFETY: the allocation never moves or changes protection, and T supplies the
// same ownership/concurrent-access restrictions as an ordinary shared slice.
unsafe impl<T: Send> Send for SparseArray<T> {}
unsafe impl<T: Sync> Sync for SparseArray<T> {}
impl<T> SparseArray<T> {
    /// Reserve a fixed, initially zero-filled array without touching its pages.
    ///
    /// # Safety
    /// T must have a valid all-zero representation, alignment at most a host
    /// page, and require no destructor. Mutations use T's interior mutability.
    pub unsafe fn new(machine: &Machine, entries: usize) -> Self {
        let bytes = entries
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes != 0)
            .expect("nonempty sparse array size");
        assert!(std::mem::align_of::<T>() <= 4096);
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(
            pointer != libc::MAP_FAILED,
            "page table reservation failed: {}",
            std::io::Error::last_os_error()
        );
        unsafe {
            libc::madvise(pointer, bytes, libc::MADV_DONTDUMP);
            libc::madvise(pointer, bytes, libc::MADV_NOHUGEPAGE);
        }
        machine.reserve_page_table(bytes as u64);
        Self {
            machine: machine.clone(),
            base: NonNull::new(pointer.cast()).expect("page table base"),
            entries,
            bytes,
        }
    }
    /// # Safety
    /// index must be smaller than the reserved entry count.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.entries);
        unsafe { &*self.base.as_ptr().add(index) }
    }
}
impl<T> Index<usize> for SparseArray<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        assert!(index < self.entries, "page table index exceeds reservation");
        unsafe { self.get_unchecked(index) }
    }
}
impl<T> Drop for SparseArray<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.bytes);
        }
        self.machine.release_page_table(self.bytes as u64);
    }
}
