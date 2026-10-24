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
pub fn mapping(allocation: &Allocation) -> std::sync::Arc<nixe_gpu::physical::PhysicalMapping> {
    nixe_gpu::physical::PhysicalMapping::new(&ranges(allocation)).unwrap()
}
pub fn write(allocation: &Allocation, offset: u64, bytes: &[u8]) -> Result<(), MemoryError> {
    let range = AllocationRange::new(allocation.pin(), offset, bytes.len() as u64)?;
    range
        .allocation()
        .machine()
        .write_resolved(std::slice::from_ref(&range), bytes)
}
pub fn access(
    machine: &Machine,
) -> impl FnMut(
    Vec<AllocationRange>,
    nixe_memory::AccessOperation,
) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>>
+ use<> {
    let machine = machine.clone();
    move |ranges, operation| {
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(std::sync::Arc::new)
    }
}
pub fn apply(
    ranges: Vec<AllocationRange>,
    operation: nixe_memory::AccessOperation,
) -> Result<nixe_memory::AccessResult, std::sync::Arc<MemoryError>> {
    let memory = ranges
        .first()
        .expect("nonempty RAM operation")
        .allocation()
        .machine()
        .clone();
    memory
        .apply_resolved_access(&ranges, &operation)
        .map_err(std::sync::Arc::new)
}

/// Run a real command future whose fixture has no external dependency.
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
