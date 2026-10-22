/// One read-only snapshot of the allocator's guest ownership and kernel heap.
/// Aliases, retained native pins and duplicate handles never add another charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemoryAccounting {
    limit: u64,
    used: u64,
    heap: u64,
    system_reservation: u64,
}
impl ProcessMemoryAccounting {
    pub(crate) fn new(limit: u64, used: u64, heap: u64, system_reservation: u64) -> Self {
        Self {
            limit,
            used,
            heap,
            system_reservation,
        }
    }
    pub const fn total_user_physical_memory_size(self) -> u64 {
        self.limit + self.system_reservation
    }
    pub const fn used_user_physical_memory_size(self) -> u64 {
        self.used + self.system_reservation
    }
    pub const fn total_non_system_user_physical_memory_size(self) -> u64 {
        self.limit
    }
    pub const fn used_non_system_user_physical_memory_size(self) -> u64 {
        self.used
    }
    pub const fn total_system_resource_size(self) -> u64 {
        self.system_reservation
    }
    pub const fn used_system_resource_size(self) -> u64 {
        0
    }
    pub const fn heap_size(self) -> u64 {
        self.heap
    }
}
