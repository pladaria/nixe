use super::*;
use crate::lifetime::unit::EdgeKind;
use nixe_cpu::{
    memory::{MemoryPermissions, ProcessMemory},
    platform::TargetPlatform,
};
use nixe_memory::{AddressSpaceId, DirectBackendPolicy, GuestPhysicalPageId};

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const PC: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);

mod budget;
mod capacity;
mod completion;
mod execution;
mod fallback;
mod lifecycle;
mod rsb;
mod shutdown;
mod worker;

fn cpu() -> ProcessCpuContext {
    ProcessCpuContext::new(TargetPlatform::Switch1, SPACE)
}

fn memory(policy: DirectBackendPolicy) -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    memory
        .initialize_ram(page, 0, &[0x1f, 0x20, 0x03, 0xd5, 0x20, 0, 0x20, 0xd4])
        .unwrap(); // NOP; BRK #1.
    assert!(memory.map_page(SPACE, PC, page, MemoryPermissions::READ_EXECUTE));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, policy)
        .unwrap();
    Arc::new(memory)
}

fn breakpoint(thread: &mut JitThread, worker: &mut NativeWorker, immediate: u16) {
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            worker,
            &mut state,
            PollBudget::new(4096, 10).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap()
    else {
        panic!("expected native exit")
    };
    assert_eq!(guest.kind, EdgeKind::Breakpoint(immediate));
    assert_eq!(guest.pc, PC.checked_add(4).unwrap());
    assert_eq!(budget.slice_remaining, 9);
}

#[test]
fn demanded_kernel_executes_native_code_and_recompiles_after_bound_memory_mutation() {
    let mut worker = NativeWorker::default();
    let memory = memory(DirectBackendPolicy::Required);
    let process = Arc::new(JitProcess::new(cpu(), memory.clone()).unwrap());
    let mut thread = JitThread::new(process.clone()).unwrap();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let (exit, budget) = thread
        .invoke(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            PollBudget::new(4096, 10).unwrap(),
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(exit.is_none());
    assert_eq!(budget.slice_remaining, 10);
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    breakpoint(&mut thread, &mut worker, 1);
    // This requests the real bound coordinator, with no retained invocation or
    // memory lease. Neither the old frame nor its native entry can survive it.
    memory
        .overwrite_mapped_ram(
            SPACE,
            PC.checked_add(4).unwrap(),
            &0xd4200120_u32.to_le_bytes(),
        )
        .unwrap();
    state.set_pc(PC.get());
    assert!(
        thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(4096, 10).unwrap(),
                &VcpuEventState::default()
            )
            .unwrap()
            .0
            .is_none()
    );
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    breakpoint(&mut thread, &mut worker, 9);
}

#[test]
fn vcpus_share_published_code_but_register_fault_stacks_on_the_executing_worker() {
    let mut worker = NativeWorker::default();
    let process = Arc::new(JitProcess::new(cpu(), memory(DirectBackendPolicy::Required)).unwrap());
    let mut first = JitThread::new(process.clone()).unwrap();
    let mut second = JitThread::new(process).unwrap();
    assert!(matches!(first.demand(PC).unwrap(), Demand::Ready));
    // The second demand reuses the exact-key payload; it must not need a fresh
    // capture or compiler identity and can move before native registration.
    assert!(matches!(second.demand(PC).unwrap(), Demand::Ready));
    std::thread::spawn(move || breakpoint(&mut second, &mut NativeWorker::default(), 1))
        .join()
        .unwrap();
    breakpoint(&mut first, &mut worker, 1);
}

#[test]
fn guest_return_predictions_survive_native_slices_and_vcpu_migration() {
    let process = Arc::new(JitProcess::new(cpu(), memory(DirectBackendPolicy::Required)).unwrap());
    let mut first = JitThread::new(process.clone()).unwrap();
    let mut second = JitThread::new(process.clone()).unwrap();
    assert!(matches!(first.demand(PC).unwrap(), Demand::Ready));
    let key = first.key(PC.checked_add(64).unwrap()).unwrap();
    let mut returns = crate::ReturnStack {
        entries: [crate::rsb::Continuation::from(key); crate::rsb::CAPACITY],
        head: 0,
        depth: 16,
    };
    let original = returns.clone();
    let mut worker = NativeWorker::default();
    for index in [0, 1, 0] {
        let thread = if index == 0 { &mut first } else { &mut second };
        let mut state = A64State::default();
        state.set_pc(PC.get()); // NOP; BRK, neither modifies the return stack.
        let (Some(invocation::Exit::Native { guest, .. }), budget) = thread
            .invoke(
                &mut returns,
                &mut worker,
                &mut state,
                PollBudget::new(4096, 10).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!("expected native breakpoint")
        };
        assert_eq!(guest.kind, EdgeKind::Breakpoint(1));
        assert_eq!(budget.slice_remaining, 9);
        assert_eq!(returns, original);
    }
    assert!(process.lifetime.try_shutdown().unwrap());
    assert_eq!(returns, original); // Scalar guest keys own no native storage.
}

#[test]
fn first_word_fetch_faults_are_owned_and_not_published() {
    let mut worker = NativeWorker::default();
    let memory = memory(DirectBackendPolicy::Required);
    let process = Arc::new(JitProcess::new(cpu(), memory.clone()).unwrap());
    let mut thread = JitThread::new(process).unwrap();
    for (address, reason) in [
        (0x2000, InstructionFetchFaultReason::Unmapped),
        (0x1001, InstructionFetchFaultReason::Misaligned),
    ] {
        let Demand::FetchFault(fault) = thread.demand(GuestVirtualAddress::new(address)).unwrap()
        else {
            panic!("expected exact first-word fault")
        };
        assert_eq!(fault.reason, reason);
        assert_eq!(fault.address, GuestVirtualAddress::new(address));
    }
    memory
        .set_permissions(SPACE, PC, 4096, MemoryPermissions::READ)
        .unwrap();
    let Demand::FetchFault(fault) = thread.demand(PC).unwrap() else {
        panic!()
    };
    assert_eq!(
        fault.reason,
        InstructionFetchFaultReason::ExecutePermissionDenied
    );
    memory
        .set_permissions(SPACE, PC, 4096, MemoryPermissions::READ_EXECUTE)
        .unwrap();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    breakpoint(&mut thread, &mut worker, 1);
}

#[test]
fn process_binding_rejects_checked_active_or_previously_owned_memory_without_cycles() {
    let checked = memory(DirectBackendPolicy::Disabled);
    assert!(matches!(
        JitProcess::new(cpu(), checked),
        Err(Error {
            kind: crate::jit_error::Kind::Unsupported,
            ..
        })
    ));
    let memory = memory(DirectBackendPolicy::Required);
    let lease = memory.acquire_execution_lease();
    assert!(
        JitProcess::new(cpu(), memory.clone())
            .err()
            .unwrap()
            .detail
            .contains("idle, unbound gate")
    );
    drop(lease);
    let process = Arc::new(JitProcess::new(cpu(), memory.clone()).unwrap());
    assert!(
        JitProcess::new(cpu(), memory.clone())
            .err()
            .unwrap()
            .detail
            .contains("idle, unbound gate")
    );
    let weak_process = Arc::downgrade(&process);
    let weak_memory = Arc::downgrade(&memory);
    let weak_lifetime = Arc::downgrade(&process.lifetime);
    drop(process);
    assert!(weak_process.upgrade().is_none());
    drop(memory);
    assert!(weak_memory.upgrade().is_none());
    assert!(weak_lifetime.upgrade().is_none());
}
