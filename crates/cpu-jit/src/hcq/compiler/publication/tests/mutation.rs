use super::lifecycle::{demand, payload, promote_at, run, run_to};
use super::negative::reshape;
use super::*;
use crate::lifetime::Error;
use nixe_cpu::memory::{CacheMaintenanceKind, ProcessMemory};
use nixe_memory::MEMORY_INVALIDATION_CAPACITY;

fn publish(frozen: &Frozen<'_, '_>, memory: &ExecutionMemory) -> Result<UnitHandle, Failure> {
    Compiler::new(host(), 0x10000).unwrap().publish(
        &mut Context::new(),
        &mut FunctionBuilderContext::new(),
        frozen,
        memory,
    )
}

#[test]
fn alias_ic_invalidates_real_replacement_and_baselines_but_preserves_unrelated_compilation() {
    for pending_cutover in [false, true] {
        let (process, mut memory, mut reader) = setup();
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x8000),
            GuestPhysicalPageId::new(2),
            MemoryPermissions::READ_WRITE_EXECUTE,
        ));
        demand(&process, &memory, &mut reader, 0x8000);
        let alias = promote_at(&process, &memory, &mut reader, 0x8000);
        let predecessor = promote_at(&process, &memory, &mut reader, 0x2000);
        let pins = [alias, predecessor].map(|handle| process.snapshot(handle).unwrap());
        // Warm direct, indirect and return ingress before replacement.
        for _ in 0..2 {
            for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
                run(&process, &memory, &mut reader, pc, expected);
            }
        }
        let unrelated_work = work_at(&process, &mut reader, 0x7000);
        let unrelated = unrelated_work
            .reserve_candidate(Graph::discover(&unrelated_work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let baseline = frozen
            .graph()
            .units
            .iter()
            .find(|unit| unit.instructions.get(0).unwrap().key.block_key() == key(0x2000))
            .unwrap()
            .registered_handle()
            .unwrap();
        let successor = publish(&frozen, &memory).unwrap();
        if !pending_cutover {
            process.try_service_links().unwrap();
        }
        let cursor = memory.invalidation_cursor();
        // Change an interior instruction through the other virtual address:
        // BR X2 -> BRK #9. A data write alone does not publish new instructions.
        memory
            .write(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x8004),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0xd4200120),
            )
            .unwrap();
        assert_eq!(memory.invalidation_cursor(), cursor);
        if pending_cutover {
            assert!(matches!(process.snapshot(successor), Err(Error::Closed)));
            assert!(matches!(process.snapshot(alias), Err(Error::Closed)));
        } else {
            assert!(process.snapshot(successor).is_ok());
            assert!(process.snapshot(alias).is_ok());
            run(&process, &memory, &mut reader, 0x3000, 3);
            run(&process, &memory, &mut reader, 0x8000, 2);
        }
        memory
            .maintain_cache(
                AddressSpaceId::new(1),
                CacheMaintenanceKind::InstructionInvalidate,
                Some(GuestVirtualAddress::new(0x8004)),
            )
            .unwrap();
        assert!(memory.invalidation_cursor() > cursor);
        process.try_service_links().unwrap();
        for handle in [alias, predecessor, successor, baseline] {
            assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
        }
        assert!(payload(&mut reader, 0x2000).is_none());
        assert!(payload(&mut reader, 0x8000).is_none());
        assert!(payload(&mut reader, 0x1000).unwrap().hcq().is_none());
        // Exact page invalidation must not cancel a worker on the return page,
        // even though both a maintenance epoch and the memory cursor advanced.
        unrelated.check().unwrap();
        publish(&unrelated, &memory).unwrap();
        drop(unrelated);
        drop(unrelated_work);
        drop(frozen);
        drop(work);
        drop(pins);
        process.try_service_links().unwrap();
        process.reclaim_units().unwrap();
        for pc in [0x2000, 0x8000] {
            demand(&process, &memory, &mut reader, pc);
        }
        for _ in 0..2 {
            for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
                run_to(&process, &memory, &mut reader, pc, expected, (0x2004, 9));
            }
            run_to(&process, &memory, &mut reader, 0x8000, 2, (0x8004, 9));
        }
        assert!(process.try_shutdown().unwrap());
        assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    }
}

#[test]
fn real_memory_history_loss_cancels_pending_replacement_all_baselines_and_unrelated_work() {
    let (process, mut memory, mut reader) = setup();
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(8)));
    assert!(memory.map_page(
        AddressSpaceId::new(1),
        GuestVirtualAddress::new(0x8000),
        GuestPhysicalPageId::new(8),
        MemoryPermissions::READ_WRITE,
    ));
    let predecessor = promote_at(&process, &memory, &mut reader, 0x2000);
    let pin = process.snapshot(predecessor).unwrap();
    for (pc, expected) in [(0x3000, 3), (0x6000, 3), (0x7000, 2)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    let unrelated_work = work_at(&process, &mut reader, 0x6000);
    let unrelated = unrelated_work
        .reserve_candidate(Graph::discover(&unrelated_work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let baselines: Vec<_> = frozen
        .graph()
        .units
        .iter()
        .map(|unit| unit.registered_handle().unwrap())
        .collect();
    let mut cursor = memory.invalidation_cursor();
    // Overflow the real bounded log with coordinated mutations on an unrelated,
    // non-code page. No fabricated HistoryLost result or unsafe memory producer.
    for index in 0..=MEMORY_INVALIDATION_CAPACITY {
        memory
            .set_permissions(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x8000),
                4096,
                if index % 2 == 0 {
                    MemoryPermissions::READ
                } else {
                    MemoryPermissions::READ_WRITE
                },
            )
            .unwrap();
    }
    frozen.check().unwrap();
    unrelated.check().unwrap();
    assert!(process.snapshot(predecessor).is_ok());
    let successor = publish(&frozen, &memory).unwrap();
    // Leave cutover pending: the overrun consumer must also find predecessors
    // which are no longer the preferred dispatch owner.
    let observed = memory.invalidation_cursor();
    assert!(matches!(
        memory.read_invalidations_since(cursor, &mut Vec::new()),
        Err(MemoryInvalidationError::HistoryLost { latest, .. }) if latest == observed
    ));
    process
        .consume_memory_invalidations(&memory, &mut cursor)
        .unwrap();
    assert_eq!(cursor, observed);
    process.try_service_links().unwrap();
    for handle in baselines.into_iter().chain([predecessor, successor]) {
        assert!(matches!(process.snapshot(handle), Err(Error::StaleUnit)));
    }
    for pc in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000] {
        assert!(payload(&mut reader, pc).is_none());
    }
    assert_eq!(unrelated.check(), Err(Error::StalePublication));
    assert!(matches!(
        publish(&unrelated, &memory),
        Err(Failure::Cancelled)
    ));
    drop(unrelated);
    drop(unrelated_work);
    drop(frozen);
    drop(work);
    drop(pin);
    process.reclaim_units().unwrap();
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
    for pc in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000] {
        demand(&process, &memory, &mut reader, pc);
    }
    for (pc, expected) in [(0x3000, 3), (0x4000, 2), (0x6000, 3), (0x7000, 2)] {
        run(&process, &memory, &mut reader, pc, expected);
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}
