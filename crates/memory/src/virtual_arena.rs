//! Per-process address reservations alias the machine's single physical file.
use crate::{ArenaOffset, GuestVirtualAddress, HostAddress, Machine, MemoryError, PAGE_SIZE};
use std::{ffi::c_void, ptr::NonNull};

const ALIGNMENT: usize = 2 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddressWidth {
    Bits32,
    Bits36,
    Bits39,
}
impl AddressWidth {
    pub const fn bits(self) -> u32 {
        match self {
            Self::Bits32 => 32,
            Self::Bits36 => 36,
            Self::Bits39 => 39,
        }
    }
    pub const fn bytes(self) -> u64 {
        1 << self.bits()
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MemoryPermissions(u8);
impl MemoryPermissions {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1);
    pub const WRITE: Self = Self(2);
    pub const EXECUTE: Self = Self(4);
    pub const READ_WRITE: Self = Self(3);
    pub const READ_EXECUTE: Self = Self(5);
    pub const ALL: Self = Self(7);
    pub const fn bits(self) -> u8 {
        self.0
    }
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !7 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }
    pub const fn contains(self, permissions: Self) -> bool {
        self.0 & permissions.0 == permissions.0
    }
    pub const fn intersection(self, permissions: Self) -> Self {
        Self(self.0 & permissions.0)
    }
    fn protection(self) -> i32 {
        // This is the JIT's data arena, not executable host code. Linux/x86
        // cannot enforce guest write-only or execute-only data permissions:
        // keep them inaccessible here and let checked callbacks enforce them.
        if !self.contains(Self::READ) {
            return libc::PROT_NONE;
        }
        libc::PROT_READ
            | if self.contains(Self::WRITE) {
                libc::PROT_WRITE
            } else {
                0
            }
    }
}
/// Low-level reservation. Mapping writers must establish address-space
/// quiescence and span exclusion before changing these OS mappings.
pub struct VirtualArena {
    machine: Machine,
    base: NonNull<u8>,
    width: AddressWidth,
}
// SAFETY: no borrowed RAM references are returned. Mapping mutations require
// external quiescence; checked/native accesses use their own admission contract.
unsafe impl Send for VirtualArena {}
unsafe impl Sync for VirtualArena {}
impl VirtualArena {
    pub fn new(machine: Machine, width: AddressWidth) -> Result<Self, MemoryError> {
        let bytes = usize::try_from(width.bytes())
            .map_err(|_| MemoryError::Invalid("host cannot reserve this process address width"))?;
        let guard = PAGE_SIZE as usize;
        let reserve = bytes
            .checked_add(ALIGNMENT + 2 * guard)
            .ok_or(MemoryError::Invalid("virtual reservation overflow"))?;
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserve,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let base = (raw as usize + guard + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let start = base - guard;
        let size = bytes + 2 * guard;
        let prefix = start - raw as usize;
        let suffix = reserve - prefix - size;
        unsafe {
            if prefix != 0 {
                libc::munmap(raw, prefix);
            }
            if suffix != 0 {
                libc::munmap((start + size) as *mut c_void, suffix);
            }
        }
        machine.reserve_virtual(size as u64);
        let arena = Self {
            machine,
            base: NonNull::new(base as *mut u8).expect("nonzero virtual reservation"),
            width,
        };
        if unsafe { libc::madvise(start as *mut c_void, size, libc::MADV_DONTDUMP) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(arena)
    }
    pub fn machine(&self) -> &Machine {
        &self.machine
    }
    pub fn width(&self) -> AddressWidth {
        self.width
    }
    pub fn host_address(&self, address: GuestVirtualAddress, bytes: u64) -> Option<HostAddress> {
        (address.get().checked_add(bytes)? <= self.width.bytes()).then(|| {
            HostAddress::from_pointer(unsafe { self.base.as_ptr().add(address.get() as usize) })
        })
    }
    fn page_range(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<HostAddress, MemoryError> {
        if bytes == 0 || !bytes.is_multiple_of(PAGE_SIZE) || !start.get().is_multiple_of(PAGE_SIZE)
        {
            return Err(MemoryError::Invalid(
                "virtual mapping must cover whole pages",
            ));
        }
        self.host_address(start, bytes).ok_or(MemoryError::Invalid(
            "virtual mapping exceeds the process reservation",
        ))
    }
    /// # Safety
    /// Every executing core is quiescent and checked access guards are drained.
    /// The caller retains the physical allocation pins until the mapping is removed.
    pub unsafe fn map(
        &self,
        start: GuestVirtualAddress,
        physical: ArenaOffset,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let host = self.page_range(start, bytes)?;
        if !physical.get().is_multiple_of(PAGE_SIZE)
            || self.machine.arena().host_address(physical, bytes).is_none()
        {
            return Err(MemoryError::Invalid("physical mapping exceeds machine RAM"));
        }
        let mapped = unsafe {
            libc::mmap(
                host.as_ptr().cast(),
                bytes as usize,
                permissions.protection(),
                libc::MAP_SHARED | libc::MAP_FIXED,
                self.machine.arena().descriptor(),
                physical.get() as libc::off_t,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        if unsafe { libc::madvise(mapped, bytes as usize, libc::MADV_DONTDUMP) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
    /// # Safety
    /// Every executing core is quiescent and checked access guards are drained.
    pub unsafe fn unmap(&self, start: GuestVirtualAddress, bytes: u64) -> Result<(), MemoryError> {
        let host = self.page_range(start, bytes)?;
        // Keep the reservation: removing an alias never returns a hole that an
        // unrelated host mapping could occupy before this process remaps it.
        let mapped = unsafe {
            libc::mmap(
                host.as_ptr().cast(),
                bytes as usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        if unsafe { libc::madvise(mapped, bytes as usize, libc::MADV_DONTDUMP) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
    /// # Safety
    /// The caller has quiesced cores/accesses and validated live mappings. These
    /// permissions are the intersection of guest permissions and tracking state.
    pub unsafe fn protect(
        &self,
        start: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let host = self.page_range(start, bytes)?;
        if unsafe {
            libc::mprotect(
                host.as_ptr().cast(),
                bytes as usize,
                permissions.protection(),
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
}
impl Drop for VirtualArena {
    fn drop(&mut self) {
        let guard = PAGE_SIZE as usize;
        self.machine
            .release_virtual(self.width.bytes() + 2 * PAGE_SIZE);
        unsafe {
            libc::munmap(
                self.base.as_ptr().sub(guard).cast(),
                self.width.bytes() as usize + 2 * guard,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_process_reservations_share_physical_bytes_and_keep_distinct_virtual_mappings() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let pin = allocation.pin();
        let a = VirtualArena::new(machine.clone(), AddressWidth::Bits39).unwrap();
        let b = VirtualArena::new(machine, AddressWidth::Bits32).unwrap();
        let at = GuestVirtualAddress::new(0x71_0000_0000);
        let bt = GuestVirtualAddress::new(0x9000);
        let pa = a.host_address(at, PAGE_SIZE).unwrap().as_ptr();
        let pb = b.host_address(bt, PAGE_SIZE).unwrap().as_ptr();
        assert_ne!(pa, pb);
        assert_eq!(
            a.host_address(GuestVirtualAddress::new(0), 0)
                .unwrap()
                .as_ptr() as usize
                % ALIGNMENT,
            0
        );
        unsafe {
            a.map(
                at,
                pin.physical_extents(0, pin.bytes()).next().unwrap().start,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
            b.map(
                bt,
                pin.physical_extents(0, pin.bytes()).next().unwrap().start,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
            pa.add(3).write(91);
            assert_eq!(pb.add(3).read(), 91);
            a.unmap(at, PAGE_SIZE).unwrap();
            pb.add(3).write(42);
            a.map(
                at,
                pin.physical_extents(0, pin.bytes()).next().unwrap().start,
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
            assert_eq!(pa.add(3).read(), 42);
        }
        drop(a);
        drop(b);
        drop(pin);
        drop(allocation);
    }
}
