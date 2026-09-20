use super::*;
use crate::abi::{FpSpecialization, HostAbi, PollBudget};
use crate::lcq::{Compilation, compiler::Compiler};
use nixe_cpu::{
    memory::{
        ExecutionMemory, InstructionMemory, MemoryAliasRequest, MemoryAttributes,
        MemoryMappingProperties, MemoryMappingPurpose, MemoryPermissions,
        MemoryProtectionErrorReason, ProcessMemory,
    },
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::A64State,
};
use nixe_memory::{
    AddressSpaceId, DirectBackendPolicy, GuestPhysicalPageId, GuestVirtualAddress,
    MemoryInvalidationSource,
};
use std::time::Duration;

mod atomics;
mod background;
mod cache;
mod capture;
mod data_cache;
mod device;
mod host_writes;
mod initialization;
mod mmio;
mod publication;
mod stream;
mod tracking;
mod writes;

const SPACE: AddressSpaceId = AddressSpaceId::new(1);

fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}

fn fixture() -> (Arc<Lifetime>, ExecutionMemory) {
    let process = Arc::new(Lifetime::new(Cache::new().unwrap()).unwrap());
    let mut memory = ExecutionMemory::new();
    for page in 1..=2 {
        let id = GuestPhysicalPageId::new(page);
        assert!(memory.add_ram_page(id));
        let bytes: Vec<_> = [0xf9400020_u32, 0xd4200000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        memory.initialize_ram(id, 0, &bytes).unwrap(); // LDR X0, [X1]; BRK
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(page * 4096),
            id,
            MemoryPermissions::READ_EXECUTE
        ));
    }
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, DirectBackendPolicy::Required)
        .unwrap();
    memory.set_mutation_observer(process.clone()).unwrap();
    (process, memory)
}

fn compiler() -> Compiler {
    Compiler::for_arena(
        if cfg!(target_arch = "x86_64") {
            HostAbi::X86_64
        } else {
            HostAbi::Aarch64
        },
        0x10000,
    )
    .unwrap()
}

fn publish(process: &Arc<Lifetime>, memory: &ExecutionMemory, pc: u64) -> unit::UnitHandle {
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(pc)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, memory).unwrap();
    compiler()
        .publish(captured, process, &process.cache, memory)
        .unwrap()
}

#[test]
fn permission_mutation_drains_real_lcq_fault_reader_before_becoming_visible() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let other = publish(&process, &memory, 0x2000);
    let snapshot = process.snapshot(old).unwrap();
    let cursor = memory.invalidation_cursor();
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let lease = memory.acquire_execution_lease();
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| {
            memory.set_permissions(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                4096,
                MemoryPermissions::READ,
            )
        });
        {
            let state = process.lock();
            let (state, timeout) = process
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |state| {
                    state.phase == Phase::Open
                })
                .unwrap();
            assert!(!timeout.timed_out());
            assert_eq!(state.phase, Phase::Closing);
        }
        assert_eq!(fault.unit.id, snapshot.id);
        assert!(
            memory
                .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
                .is_ok()
        );
        assert_eq!(memory.invalidation_cursor(), cursor);
        drop(invocation);
        drop(lease); // The writer never waits for an epoch/lease it owns itself.
        worker.join().unwrap().unwrap();
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert!(memory.invalidation_cursor() > cursor);
    assert!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .is_err()
    );
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(other).is_ok());
    assert!(
        unsafe { reader.admit(&mut frame, key(0x1000)) }
            .unwrap()
            .is_none()
    );
    assert_eq!(process.reclaim_units().unwrap(), 0);
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn capture_does_not_cancel_itself_but_a_mapping_mutation_cancels_its_publication() {
    let (process, memory) = fixture();
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, &memory).unwrap();
    compilation.claim.validate().unwrap();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap();
    assert_eq!(process.lock().phase, Phase::Open);
    assert_eq!(compilation.claim.validate(), Err(Error::StalePublication));
    assert!(matches!(
        compiler().publish(compilation, &process, &process.cache, &memory),
        Err(crate::lcq::compiler::PublishError::Lifetime(
            Error::StalePublication
        ))
    ));
}

#[test]
fn actual_attribute_change_and_failed_mapping_preflight_release_the_stop() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    memory
        .set_attributes(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryAttributes::PERMISSION_LOCKED,
            MemoryAttributes::PERMISSION_LOCKED,
        )
        .unwrap();
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    let cursor = memory.invalidation_cursor();
    let fresh = publish(&process, &memory, 0x1000);
    assert!(
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                4096,
                MemoryPermissions::READ
            )
            .is_err()
    );
    assert_eq!(process.lock().phase, Phase::Open);
    assert_eq!(memory.invalidation_cursor(), cursor);
    // A failed preflight may discard derived code, never change guest memory.
    assert!(matches!(process.snapshot(fresh), Err(Error::StaleUnit)));
    assert!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .is_ok()
    );
    publish(&process, &memory, 0x1000);
}

#[test]
fn mutation_hold_prevents_premature_acknowledgement_and_coalesces_with_another_hold() {
    let (process, _memory) = fixture();
    let first = process.clone().begin(&[]).unwrap();
    let second = process.clone().begin(&[]).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert!(!transition.try_reopen().unwrap());
    drop(transition);
    drop(first);
    assert_eq!(process.lock().phase, Phase::Closed);
    drop(second);
    assert_eq!(process.lock().phase, Phase::Open);
    assert_eq!(process.lock().memory_mutations, 0);
}

#[test]
fn memory_request_joins_an_existing_owner_without_requiring_it_to_relinquish_the_stop() {
    let (process, memory) = fixture();
    publish(&process, &memory, 0x1000);
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    std::thread::scope(|scope| {
        let (send, receive) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let process = &process;
        let worker = scope.spawn(move || {
            let hold = process
                .clone()
                .begin(&[MemoryInvalidationKind::Mapping {
                    address_space: SPACE,
                    start: GuestVirtualAddress::new(0x1000),
                    size: 4096,
                }])
                .unwrap();
            send.send(()).unwrap();
            released.recv().unwrap();
            drop(hold);
        });
        {
            let state = process.lock();
            let (_state, timeout) = process
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |state| {
                    state.pending[Reason::MappingChange as usize].is_none()
                })
                .unwrap();
            assert!(!timeout.timed_out());
        }
        transition.wait_closed().unwrap();
        assert_eq!(transition.drain_retirements().unwrap(), 1);
        receive.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(
            transition.batch().unwrap().complete(),
            Err(Error::MaintenancePending)
        );
        release.send(()).unwrap();
        worker.join().unwrap();
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
    });
}

#[test]
fn unwinding_memory_work_disables_admission_instead_of_reopening() {
    let (process, _memory) = fixture();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _hold = process.clone().begin(&[]).unwrap();
        panic!("mutation failed after entering the authority");
    }));
    assert!(result.is_err());
    assert!(matches!(
        process.reserve(key(0x1000)),
        Err(Error::InvalidUnit(_))
    ));
    assert_eq!(process.lock().memory_mutations, 0);
    assert_ne!(process.lock().phase, Phase::Open);
}

#[test]
fn alias_creation_and_removal_unlink_both_virtual_ranges_but_keep_unrelated_code() {
    let (process, memory) = fixture();
    let old = publish(&process, &memory, 0x1000);
    let other = publish(&process, &memory, 0x2000);
    let executable = MemoryMappingProperties::new(
        MemoryPermissions::READ_EXECUTE,
        MemoryMappingPurpose::Normal,
        MemoryAttributes::NONE,
    );
    let readable = MemoryMappingProperties {
        permissions: MemoryPermissions::READ,
        ..executable
    };
    memory
        .map_alias(MemoryAliasRequest {
            address_space: SPACE,
            source: GuestVirtualAddress::new(0x1000),
            destination: GuestVirtualAddress::new(0x3000),
            size: 4096,
            source_before: executable,
            source_after: readable,
            destination_properties: executable,
        })
        .unwrap();
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(other).is_ok());
    assert_eq!(process.lock().phase, Phase::Open);
    let alias = publish(&process, &memory, 0x3000);
    memory
        .unmap_alias(MemoryAliasRequest {
            address_space: SPACE,
            source: GuestVirtualAddress::new(0x1000),
            destination: GuestVirtualAddress::new(0x3000),
            size: 4096,
            source_before: readable,
            source_after: executable,
            destination_properties: executable,
        })
        .unwrap();
    assert!(matches!(process.snapshot(alias), Err(Error::StaleUnit)));
    assert!(process.snapshot(other).is_ok());
    assert!(
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(0x3000))
            .is_none()
    );
    publish(&process, &memory, 0x1000);
}

#[test]
fn resize_only_invalidates_the_changed_tail_and_replacement_uses_fresh_code() {
    let (process, memory) = fixture();
    let first = publish(&process, &memory, 0x1000);
    let tail = publish(&process, &memory, 0x2000);
    let old = process.snapshot(tail).unwrap();
    memory
        .resize_zeroed_mapping(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            8192,
            4096,
            MemoryPermissions::READ_EXECUTE,
            MemoryMappingPurpose::Normal,
        )
        .unwrap();
    assert!(process.snapshot(first).is_ok());
    assert!(matches!(process.snapshot(tail), Err(Error::StaleUnit)));
    assert!(
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(0x2000))
            .is_none()
    );
    memory
        .resize_zeroed_mapping(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            8192,
            MemoryPermissions::READ_EXECUTE,
            MemoryMappingPurpose::Normal,
        )
        .unwrap();
    let fresh = publish(&process, &memory, 0x2000);
    let new = process.snapshot(fresh).unwrap();
    assert_ne!(new.id, old.id);
    assert_ne!(new.dependencies[0], old.dependencies[0]);
    assert_eq!(new.instructions[0].bits, 0);
    assert!(process.snapshot(first).is_ok());
}

#[test]
fn coordinator_failure_rejects_the_memory_operation_with_its_actual_diagnostic() {
    let (process, memory) = fixture();
    let cursor = memory.invalidation_cursor();
    process.fail(
        &mut process.lock(),
        Error::Capacity("injected coordinator failure"),
    );
    let error = memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap_err();
    assert!(
        matches!(error.reason, MemoryProtectionErrorReason::ExecutionMutation(ExecutionMutationError(detail)) if detail.contains("injected coordinator failure"))
    );
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert!(
        memory
            .fetch32(SPACE, GuestVirtualAddress::new(0x1000))
            .is_ok()
    );
    assert!(!memory.mapping_mutation_pending());
    assert_eq!(process.lock().memory_mutations, 0);
}
