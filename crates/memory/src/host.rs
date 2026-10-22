//! Stable shared backing; virtual aliases map this same file, never private RAM.
use crate::{ArenaOffset, DEFAULT_RAM_BYTES, PAGE_SIZE, PHYSICAL_BASE, PhysicalAddress};
use std::{
    ffi::c_void,
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    ptr::NonNull,
};

const BASE_ALIGNMENT: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct HostAddress(usize);
impl HostAddress {
    pub(crate) fn from_pointer(pointer: *mut u8) -> Self {
        Self(pointer as usize)
    }
    pub fn as_ptr(self) -> *mut u8 {
        self.0 as *mut u8
    }
    /// Read a live command word without creating a Rust reference to guest RAM.
    ///
    /// # Safety
    /// The caller retains the physical allocation, ensures four readable bytes
    /// at `offset`, and resolves device coherence on the command owner.
    pub unsafe fn read_u32(self, offset: usize) -> u32 {
        unsafe { crate::raw::read_value(self.as_ptr().add(offset), 4) as u32 }
    }
}

pub struct PhysicalArena {
    file: OwnedFd,
    base: NonNull<u8>,
    bytes: usize,
}
// SAFETY: the mapping is stable and shared. APIs expose no Rust references to
// guest bytes. Access synchronization and guest tearing belong to the audited
// instruction primitives and mapping/physical-span admission, not this owner.
unsafe impl Send for PhysicalArena {}
unsafe impl Sync for PhysicalArena {}
impl PhysicalArena {
    pub fn new(bytes: u64) -> io::Result<Self> {
        if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } != PAGE_SIZE as libc::c_long {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native guest mappings require 4096-byte host pages",
            ));
        }
        if bytes == 0 || !bytes.is_multiple_of(PAGE_SIZE) || bytes > isize::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "physical RAM capacity must be a nonzero page multiple",
            ));
        }
        let bytes = bytes as usize;
        let reserve = bytes.checked_add(BASE_ALIGNMENT).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "physical RAM reservation overflow",
            )
        })?;
        // SAFETY: the name is terminated and successful descriptors are owned once.
        let fd = unsafe { libc::memfd_create(c"nixe-physical-ram".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let file = unsafe { OwnedFd::from_raw_fd(fd) };
        if unsafe { libc::ftruncate(file.as_raw_fd(), bytes as libc::off_t) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let reserved = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if reserved == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let aligned = ((reserved as usize) + BASE_ALIGNMENT - 1) & !(BASE_ALIGNMENT - 1);
        let prefix = aligned - reserved as usize;
        let suffix = reserve - prefix - bytes;
        // SAFETY: trim only page-aligned subranges of this reservation.
        unsafe {
            if prefix != 0 {
                libc::munmap(reserved, prefix);
            }
            if suffix != 0 {
                libc::munmap((aligned + bytes) as *mut c_void, suffix);
            }
        }
        let mapped = unsafe {
            libc::mmap(
                aligned as *mut c_void,
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            let error = io::Error::last_os_error();
            unsafe {
                libc::munmap(aligned as *mut c_void, bytes);
            }
            return Err(error);
        }
        let result = Self {
            file,
            base: NonNull::new(mapped.cast()).expect("aligned physical base"),
            bytes,
        };
        if unsafe { libc::madvise(mapped, bytes, libc::MADV_DONTDUMP) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // Optional large-page advice changes neither allocation nor correctness.
        unsafe {
            libc::madvise(mapped, bytes, libc::MADV_HUGEPAGE);
        }
        Ok(result)
    }
    pub fn switch_ram() -> io::Result<Self> {
        Self::new(DEFAULT_RAM_BYTES)
    }
    /// Allocated backing-file pages, distinct from virtual reservation and RSS.
    pub fn committed_bytes(&self) -> io::Result<u64> {
        let mut status = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(self.descriptor(), status.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { status.assume_init() }.st_blocks as u64 * 512)
    }
    pub fn capacity(&self) -> u64 {
        self.bytes as u64
    }
    pub fn host_address(&self, offset: ArenaOffset, bytes: u64) -> Option<HostAddress> {
        (offset.get().checked_add(bytes)? <= self.capacity())
            .then(|| HostAddress(self.base.as_ptr() as usize + offset.get() as usize))
    }
    pub fn physical_address(&self, offset: ArenaOffset) -> Option<PhysicalAddress> {
        (offset.get() < self.capacity()).then(|| PhysicalAddress::new(PHYSICAL_BASE + offset.get()))
    }
    pub fn arena_offset(&self, address: PhysicalAddress) -> Option<ArenaOffset> {
        let offset = address.get().checked_sub(PHYSICAL_BASE)?;
        (offset < self.capacity()).then(|| ArenaOffset::new(offset))
    }
    /// # Safety
    /// The allocator exclusively owns these unpublished extents. Every previous
    /// mapping/request/native pin has been released and no new user can enter.
    pub(crate) unsafe fn clear_unpublished(&self, offset: ArenaOffset, bytes: u64) {
        let address = self
            .host_address(offset, bytes)
            .expect("allocated physical extent");
        unsafe {
            libc::memset(address.as_ptr().cast(), 0, bytes as usize);
        }
    }
    pub(crate) fn descriptor(&self) -> i32 {
        self.file.as_raw_fd()
    }
}
impl Drop for PhysicalArena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_ram_is_aligned_sparse_and_uses_a_close_on_exec_shared_file() {
        let arena = PhysicalArena::switch_ram().unwrap();
        let mut status = std::mem::MaybeUninit::<libc::stat>::uninit();
        assert_eq!(
            unsafe { libc::fstat(arena.descriptor(), status.as_mut_ptr()) },
            0
        );
        assert_eq!(
            unsafe { status.assume_init() }.st_blocks,
            0,
            "reserving RAM must not commit the 4 GiB backing"
        );
        let pointer = arena
            .host_address(ArenaOffset::new(0), PAGE_SIZE)
            .unwrap()
            .as_ptr();
        assert_eq!(pointer as usize % BASE_ALIGNMENT, 0);
        assert_eq!(arena.capacity(), 4 << 30);
        let flags = unsafe { libc::fcntl(arena.descriptor(), libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        let alias = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGE_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                arena.descriptor(),
                0,
            )
        };
        assert_ne!(alias, libc::MAP_FAILED);
        // This test has exclusive access; it creates no racing Rust reference.
        unsafe {
            assert_eq!(pointer.read(), 0);
            pointer.write(83);
            assert_eq!(alias.cast::<u8>().read(), 83);
            libc::munmap(alias, PAGE_SIZE as usize);
        }
        assert_eq!(
            arena.arena_offset(PhysicalAddress::new(PHYSICAL_BASE)),
            Some(ArenaOffset::new(0))
        );
        assert!(
            arena
                .host_address(ArenaOffset::new(arena.capacity()), 1)
                .is_none()
        );
    }
}
