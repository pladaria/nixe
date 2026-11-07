//! Canonical physical allocations for unit fixtures; no production memory backend.
use nixe_memory::{Allocation, AllocationRange, Machine, MemoryError, PAGE_SIZE};
pub fn machine() -> Machine {
    Machine::new(64 * 1024 * 1024).unwrap()
}
pub fn allocate(machine: &Machine, bytes: u64) -> Allocation {
    let bytes = bytes.next_multiple_of(PAGE_SIZE);
    machine.account(bytes).unwrap().allocate(bytes).unwrap()
}
pub fn ranges(allocation: &Allocation) -> Vec<AllocationRange> {
    let pin = allocation.pin();
    let bytes = pin.bytes();
    vec![AllocationRange::new(pin, 0, bytes).unwrap()]
}
pub fn mapping(allocation: &Allocation) -> std::sync::Arc<nixe_memory::PhysicalMapping> {
    nixe_memory::PhysicalMapping::new(&ranges(allocation)).unwrap()
}
pub fn write(allocation: &Allocation, offset: u64, bytes: &[u8]) -> Result<(), MemoryError> {
    let range = AllocationRange::new(allocation.pin(), offset, bytes.len() as u64)?;
    range
        .allocation()
        .machine()
        .write_resolved(std::slice::from_ref(&range), bytes)
}
pub struct RamAccess {
    machine: Machine,
}
impl nixe_gpu::OwnerMemory for RamAccess {
    fn invalidate_gpu_cache(&mut self) {}
    fn invalidate_regions(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }
    fn inline_memory(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }

    fn flush_region(&mut self, _: u64, _: u64) -> Result<(), std::sync::Arc<MemoryError>> {
        Ok(())
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        mut operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>> {
        self.machine
            .apply_resolved_access(&ranges, &mut operation)
            .map_err(std::sync::Arc::new)
    }
}
pub fn access(machine: &Machine) -> RamAccess {
    RamAccess {
        machine: machine.clone(),
    }
}

/// Hooks for tests of safe reads and immutable physical observations.
pub struct Access<F, G> {
    pub operation: F,
    pub flush: G,
}
impl<F, G> nixe_gpu::OwnerMemory for Access<F, G>
where
    F: FnMut(
        Vec<AllocationRange>,
        nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>>,
    G: FnMut(u64, u64) -> Result<(), std::sync::Arc<MemoryError>>,
{
    fn invalidate_gpu_cache(&mut self) {}
    fn invalidate_regions(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }
    fn inline_memory(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }

    fn flush_region(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), std::sync::Arc<MemoryError>> {
        (self.flush)(address, bytes)
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>> {
        (self.operation)(ranges, operation)
    }
}
pub struct PhysicalAccess;
impl nixe_gpu::OwnerMemory for PhysicalAccess {
    fn invalidate_gpu_cache(&mut self) {}
    fn invalidate_regions(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }
    fn inline_memory(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }

    fn flush_region(&mut self, _: u64, _: u64) -> Result<(), std::sync::Arc<MemoryError>> {
        Ok(())
    }
    fn access_memory(
        &mut self,
        ranges: Vec<AllocationRange>,
        mut operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>> {
        ranges
            .first()
            .expect("nonempty RAM operation")
            .allocation()
            .machine()
            .apply_resolved_access(&ranges, &mut operation)
            .map_err(std::sync::Arc::new)
    }
}
pub fn observed<F>(
    operation: F,
) -> Access<F, impl FnMut(u64, u64) -> Result<(), std::sync::Arc<MemoryError>>>
where
    F: FnMut(
        Vec<AllocationRange>,
        nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>>,
{
    Access {
        operation,
        flush: |_, _| Ok(()),
    }
}
pub struct SafeRead<G>(G);
impl<G: FnMut(u64, u64) -> Result<(), std::sync::Arc<MemoryError>>> nixe_gpu::OwnerMemory
    for SafeRead<G>
{
    fn invalidate_gpu_cache(&mut self) {}
    fn invalidate_regions(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }
    fn inline_memory(
        &mut self,
        _: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        Ok(())
    }

    fn flush_region(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), std::sync::Arc<MemoryError>> {
        (self.0)(address, bytes)
    }
    fn access_memory(
        &mut self,
        _: Vec<AllocationRange>,
        _: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>> {
        panic!("safe reads copy RAM directly after flushing")
    }
}
pub fn safe_read<G: FnMut(u64, u64) -> Result<(), std::sync::Arc<MemoryError>>>(
    flush: G,
) -> SafeRead<G> {
    SafeRead(flush)
}

/// Run a real command future whose fixture has no external dependency.
#[cfg(test)]
pub(crate) fn ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => panic!("fixture unexpectedly suspended"),
    }
}
