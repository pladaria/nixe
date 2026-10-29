use nixe_memory::{
    AllocationRange, GuestVirtualAddress, Machine, MemoryPermissions, capture_subrange,
};
use nixe_runtime::{
    ProcessVirtualRegion, TransferMemoryObject,
    memory::{Mapping, MemoryKind, ProcessMemory},
};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

pub(crate) fn physical(bytes: &[u8]) -> Vec<AllocationRange> {
    let size = (bytes.len() as u64).div_ceil(4096) * 4096;
    let machine = Machine::new(size).unwrap();
    let allocation = machine.account(size).unwrap().allocate(size).unwrap();
    let range = AllocationRange::new(
        allocation.pin_range(0, size).unwrap(),
        0,
        bytes.len() as u64,
    )
    .unwrap();
    machine
        .write_resolved(std::slice::from_ref(&range), bytes)
        .unwrap();
    vec![range]
}
pub(crate) fn transfer(bytes: &[u8]) -> Arc<TransferMemoryObject> {
    let size = bytes.len() as u64;
    let machine = Machine::new(size).unwrap();
    let account = machine.account(size).unwrap();
    let allocation = account.allocate(size).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, size).unwrap();
    machine.write_resolved(&[range], bytes).unwrap();
    let owner = Arc::new(
        ProcessMemory::new(
            account,
            machine.account(machine.arena().capacity()).unwrap(),
            nixe_memory::AddressWidth::Bits32,
            ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
            0,
        )
        .unwrap(),
    );
    owner
        .map(
            GuestVirtualAddress::new(0x1000),
            allocation,
            0,
            bytes.len() as u64,
            Mapping {
                kind: MemoryKind::Heap,
                permissions: MemoryPermissions::READ_WRITE,
                uncached: false,
                locked: false,
                permission_locked: false,
            },
        )
        .unwrap();
    Arc::new(
        TransferMemoryObject::new(
            owner,
            GuestVirtualAddress::new(0x1000),
            bytes.len() as u64,
            MemoryPermissions::NONE,
        )
        .unwrap(),
    )
}
pub(crate) fn read(ranges: &[AllocationRange], bytes: &mut [u8]) {
    let captured =
        nixe_memory::CapturedMemory::new(capture_subrange(ranges, 0, bytes.len() as u64).unwrap())
            .unwrap();
    let machine = captured[0].allocation().machine().clone();
    assert!(matches!(
        machine.read_captured_into(&captured, 0, bytes).unwrap(),
        nixe_memory::Access::Ready(())
    ));
}
pub(crate) fn run<F: Future>(future: F) -> F::Output {
    struct Thread(std::thread::Thread);
    impl Wake for Thread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Thread(std::thread::current())));
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub(crate) fn machine() -> Machine {
    Machine::new(16 * 1024 * 1024).unwrap()
}

pub(crate) fn process(machine: Machine) -> ProcessMemory {
    let bytes = machine.arena().capacity();
    ProcessMemory::new(
        machine.account(bytes).unwrap(),
        machine.account(bytes).unwrap(),
        nixe_memory::AddressWidth::Bits39,
        ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap()
}

pub(crate) fn kernel() -> std::sync::Arc<nixe_scheduler::SchedulerState> {
    nixe_scheduler::SchedulerState::new(
        crate::switch_1_scheduler_profile(),
        nixe_runtime::VirtualClock::new(nixe_runtime::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}
