//! Shared physical machine RAM, virtual mappings and checked memory access.
#[cfg(not(target_os = "linux"))]
compile_error!("nixe-memory requires Linux shared mappings");

mod address_space;
mod aliases;
mod captured;
mod cpu_writes;
mod device_access;
mod heap;
mod host;
mod intervals;
mod machine;
mod physical_access;
mod raw;
mod reservation;
pub mod trace;
mod tracking;
mod transition;
mod virtual_arena;
pub use address_space::{
    AddressSpace, AllocationRange, CodeFetch, MappingObservation, MappingVersion, PageEntry,
    Translation, capture_subrange,
};
pub use captured::CapturedMemory;
pub use device_access::{
    Access, AccessOperation, AccessOrdering, AccessRequestId, AccessResult, AccessResultValue,
    AtomicAccess, AtomicOperation, AtomicTransform, DeviceAccessEndpoint, DeviceAccessRequest,
    NativeReadArea, PendingAccess,
};
pub use heap::Extent;
pub use host::{HostAddress, PhysicalArena};
pub use intervals::{IntervalIndex, IntervalKey};
pub use machine::{
    Allocation, AllocationFuture, AllocationObservation, AllocationPin, Machine, MemoryAccount,
    MemoryError, MemoryStatistics, WeakAllocation,
};

pub use reservation::{DeviceWriteEpoch, NativeReservation, Reservation};
pub use tracking::{Observation, PhysicalCapture, TrackingStatistics};
pub use transition::{CodeInvalidation, CoreContext, NativeAdmission};
pub use virtual_arena::{AddressWidth, MemoryPermissions, VirtualArena};

pub const PAGE_SIZE: u64 = 4096;
pub const PHYSICAL_BASE: u64 = 0x8000_0000;
pub const DEFAULT_RAM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

macro_rules! address {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);
        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }
            pub const fn get(self) -> u64 {
                self.0
            }
            pub const fn checked_add(self, bytes: u64) -> Option<Self> {
                match self.0.checked_add(bytes) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }
        }
    };
}
address!(PhysicalAddress);
address!(ArenaOffset);
address!(GuestVirtualAddress);
address!(MachineId);
address!(AllocationId);
address!(AddressSpaceId);

impl GuestVirtualAddress {
    pub const MIN: Self = Self::new(0);
    pub const MAX: Self = Self::new(u64::MAX);
    pub const fn checked_sub(self, bytes: u64) -> Option<Self> {
        match self.0.checked_sub(bytes) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
    pub const fn checked_offset(self, bytes: i64) -> Option<Self> {
        match self.0.checked_add_signed(bytes) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
    pub const fn wrapping_add(self, bytes: u64) -> Self {
        Self(self.0.wrapping_add(bytes))
    }
    pub const fn wrapping_offset(self, bytes: i64) -> Self {
        Self(self.0.wrapping_add_signed(bytes))
    }
    pub const fn is_aligned_to(self, bytes: u64) -> bool {
        bytes.is_power_of_two() && self.0 & (bytes - 1) == 0
    }
}
impl std::fmt::Display for GuestVirtualAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}
impl std::fmt::Display for AddressSpaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "address-space=0x{:016x}", self.0)
    }
}
