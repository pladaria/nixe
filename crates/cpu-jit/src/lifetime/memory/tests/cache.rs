use super::*;
use nixe_cpu::memory::{CacheMaintenanceKind, CpuMemory, DataAccessFaultReason};

#[test]
fn instruction_cache_va_invalidation_drains_fault_readers_and_all_physical_aliases() {
    let (process, memory) = fixture();
    let properties = MemoryMappingProperties::new(
        MemoryPermissions::READ_EXECUTE,
        MemoryMappingPurpose::Normal,
        MemoryAttributes::NONE,
    );
    memory
        .map_alias(MemoryAliasRequest {
            address_space: SPACE,
            source: GuestVirtualAddress::new(0x1000),
            destination: GuestVirtualAddress::new(0x3000),
            size: 4096,
            source_before: properties,
            source_after: properties,
            destination_properties: properties,
        })
        .unwrap();
    let old = publish(&process, &memory, 0x1000);
    let alias = publish(&process, &memory, 0x3000);
    let overlap = publish(&process, &memory, 0x1004);
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
            memory.maintain_cache(
                SPACE,
                CacheMaintenanceKind::InstructionInvalidate,
                Some(GuestVirtualAddress::new(0x3004)),
            )
        });
        let (locked, timeout) = process
            .changed
            .wait_timeout_while(process.lock(), Duration::from_secs(5), |state| {
                state.phase == Phase::Open
            })
            .unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(locked.phase, Phase::Closing);
        drop(locked);
        assert_eq!(fault.unit.id, snapshot.id);
        assert_eq!(memory.invalidation_cursor(), cursor);
        drop(invocation);
        drop(lease);
        worker.join().unwrap().unwrap();
    });
    for handle in [old, alias, overlap] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    assert!(process.snapshot(other).is_ok());
    assert_eq!(process.lock().phase, Phase::Open);
    let mut records = Vec::new();
    memory
        .read_invalidations_since(cursor, &mut records)
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].kind,
        MemoryInvalidationKind::ExecutableContent {
            first: GuestPhysicalPageId::new(1),
            second: None,
        }
    );
    assert_eq!(
        records[0].origin,
        nixe_memory::MemoryInvalidationOrigin::CacheMaintenance
    );
    let new = publish(&process, &memory, 0x1000);
    assert_eq!(
        process.snapshot(new).unwrap().instructions[0].bits,
        snapshot.instructions[0].bits
    );
}

#[test]
fn instruction_cache_global_invalidation_selects_space_and_cancels_stale_capture() {
    let (process, memory) = fixture();
    let first = publish(&process, &memory, 0x1000);
    let second = publish(&process, &memory, 0x2000);
    memory
        .maintain_cache(
            AddressSpaceId::new(999),
            CacheMaintenanceKind::InstructionInvalidate,
            None,
        )
        .unwrap();
    assert!(process.snapshot(first).is_ok());
    assert!(process.snapshot(second).is_ok());
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1004)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    memory
        .maintain_cache(SPACE, CacheMaintenanceKind::InstructionInvalidate, None)
        .unwrap();
    assert_eq!(captured.claim.validate(), Err(Error::StalePublication));
    for handle in [first, second] {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    let mut records = Vec::new();
    memory
        .read_invalidations_since(cursor, &mut records)
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].kind,
        MemoryInvalidationKind::InstructionCache {
            address_space: SPACE
        }
    );
    assert_eq!(process.lock().phase, Phase::Open);
}

#[test]
fn invalid_ic_address_and_prefetch_do_not_publish_or_cancel_a_capture() {
    let (process, memory) = fixture();
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory).unwrap();
    let cursor = memory.invalidation_cursor();
    let error = memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(GuestVirtualAddress::new(0x9000)),
        )
        .unwrap_err();
    assert_eq!(error.reason, DataAccessFaultReason::Unmapped);
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionPrefetch,
            Some(GuestVirtualAddress::new(0x1000)),
        )
        .unwrap();
    assert_eq!(memory.invalidation_cursor(), cursor);
    captured.claim.validate().unwrap();
    assert!(!memory.mapping_mutation_pending());
}

#[test]
fn native_ic_completion_releases_its_own_epoch_and_advances_pc_only_on_success() {
    use crate::lcq::system::{CompletionError, RuntimeServices, complete_runtime};
    use nixe_cpu::execution::{ArchitecturalTimer, TimerSnapshot, VcpuEventState};
    struct NoTimer;
    impl ArchitecturalTimer for NoTimer {
        fn snapshot(&self) -> TimerSnapshot {
            panic!("IC does not read the timer")
        }
    }
    for mode in 0..5 {
        let (process, memory) = fixture();
        // IC IVAU, X0 or IC IALLU; BRK. This is the existing system-exit lowering, not a
        // direct invocation of the memory method in place of guest execution.
        memory
            .overwrite_mapped_ram(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                &(if mode >= 3 {
                    0xd508751f_u32
                } else {
                    0xd50b7520_u32
                })
                .to_le_bytes(),
            )
            .unwrap();
        let old = publish(&process, &memory, 0x1000);
        let mut state = A64State::default();
        state.set_pc(0x1000);
        state.general_register_storage_mut()[0] = if mode == 1 { 0x9000 } else { 0x1000 };
        let mut reader = process.register().unwrap();
        let mut monitor = nixe_cpu::exclusive::ExclusiveMonitorState::default();
        let exit = {
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
            let mut worker = nixe_cpu_direct_memory::WorkerFaultContext::register().unwrap();
            unsafe {
                crate::lcq::invocation::run(
                    &mut crate::sampling::Samples::new(),
                    &mut reader,
                    &mut frame,
                    &memory,
                    &mut worker,
                    &mut monitor,
                    key(0x1000),
                )
            }
            .unwrap()
            .unwrap()
        };
        let crate::lcq::invocation::Exit::Native { guest, .. } = exit else {
            panic!()
        };
        let unit::EdgeKind::RuntimeSystem(operation) = guest.kind else {
            panic!()
        };
        assert_eq!(guest.pc.get(), 0x1000);
        let before = state.clone();
        let cursor = memory.invalidation_cursor();
        if mode == 2 || mode == 4 {
            process.fail(
                &mut process.lock(),
                Error::Capacity("IC coordinator rejection"),
            );
        }
        let result = complete_runtime(
            operation,
            &mut state,
            &mut RuntimeServices {
                address_space: SPACE,
                memory: &memory,
                timer: &NoTimer,
                events: &VcpuEventState::default(),
                exclusive: &mut monitor,
            },
        );
        if mode == 0 || mode == 3 {
            assert_eq!(result.unwrap(), None);
            assert_eq!(state.pc(), 0x1004);
            assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
            assert!(memory.invalidation_cursor() > cursor);
        } else {
            let Err(CompletionError::Memory(fault)) = result else {
                panic!()
            };
            if mode == 1 {
                assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
            } else {
                assert!(
                    matches!(fault.reason, DataAccessFaultReason::HostBacking(detail)
                if detail.contains("IC coordinator rejection"))
                );
            }
            assert_eq!(state, before);
            assert_eq!(memory.invalidation_cursor(), cursor);
        }
        assert!(!memory.mapping_mutation_pending());
    }
}
