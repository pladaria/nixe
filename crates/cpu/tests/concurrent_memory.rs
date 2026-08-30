use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Duration;

use nixe_cpu::memory::{
    AtomicRmwKind, CacheMaintenanceKind, CpuMemory, ExecutionMemory, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryMappingPurpose, MemoryOrdering,
    MemoryPermissions, MemoryValue, ProcessMemory, SYNTHETIC_PAGE_SIZE,
};
use nixe_memory::{
    AddressSpaceId, CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
    DeviceVisibilityPoint, DeviceVisibilityRequest, GuestPhysicalPageId, GuestVirtualAddress,
    MemoryInvalidationKind, MemoryInvalidationSource, NonCpuDeviceId, VisibilityCoordinator,
    VisibilityCoordinatorError,
};

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const PRIMARY: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);

struct NoopVisibility;

impl VisibilityCoordinator for NoopVisibility {
    fn make_device_visible(
        &self,
        _request: DeviceVisibilityRequest,
        _canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        _request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        Ok(vec![0; SYNTHETIC_PAGE_SIZE].into_boxed_slice())
    }
}

fn shared_memory() -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
    assert!(memory.map_page(
        SPACE,
        PRIMARY,
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    assert!(memory.map_page(
        SPACE,
        ALIAS,
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    Arc::new(memory)
}

#[test]
fn mapping_mutation_waits_for_execution_leases_and_advances_one_epoch() {
    let memory = shared_memory();
    let initial = memory.mapping_epoch();
    let lease = memory.acquire_execution_lease();
    assert_eq!(lease.epoch(), initial);

    let worker_memory = Arc::clone(&memory);
    let (started_tx, started_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        started_tx.send(()).unwrap();
        worker_memory
            .set_permissions(
                SPACE,
                PRIMARY,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ,
            )
            .unwrap();
        finished_tx.send(()).unwrap();
    });

    started_rx.recv().unwrap();
    assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
    assert_eq!(memory.mapping_epoch(), initial);
    drop(lease);
    finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.join().unwrap();
    assert_eq!(memory.mapping_epoch().get(), initial.get() + 1);
}

#[test]
fn execution_lease_only_authorizes_its_own_memory() {
    let first = shared_memory();
    let second = shared_memory();
    let lease = first.acquire_execution_lease();

    assert!(lease.authorizes(first.as_ref()));
    assert!(!lease.authorizes(second.as_ref()));
}

#[test]
fn external_device_transition_requests_a_safepoint_and_waits_for_the_active_slice() {
    let memory = shared_memory();
    let retained = memory
        .translate_canonical_range(
            SPACE,
            PRIMARY,
            SYNTHETIC_PAGE_SIZE as u64,
            MemoryPermissions::READ,
        )
        .unwrap();
    let lease = memory.acquire_execution_lease();
    let (notified_tx, notified_rx) = mpsc::channel();
    memory.set_transition_notifier(Some(Arc::new(move || {
        let _ = notified_tx.send(());
    })));
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let declaration =
            DeviceAccessDeclaration::read(NonCpuDeviceId::new(9), DeviceVisibilityPoint::new(1));
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(NoopVisibility);
        let result = retained.prepare_device_access(declaration, coordinator);
        finished_tx.send(result).unwrap();
    });

    notified_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
    drop(lease);
    finished_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    worker.join().unwrap();
}

#[test]
fn pending_mapping_mutation_cannot_be_overtaken_by_a_new_execution_lease() {
    let memory = shared_memory();
    let initial = memory.mapping_epoch();
    let active = memory.acquire_execution_lease();

    let mutation_memory = Arc::clone(&memory);
    let mutation = thread::spawn(move || {
        mutation_memory
            .set_permissions(
                SPACE,
                PRIMARY,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ,
            )
            .unwrap();
    });
    while !memory.mapping_mutation_pending() {
        thread::yield_now();
    }

    let lease_memory = Arc::clone(&memory);
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let lease = lease_memory.acquire_execution_lease();
        acquired_tx.send(lease.epoch()).unwrap();
    });
    assert!(acquired_rx.recv_timeout(Duration::from_millis(20)).is_err());

    drop(active);
    mutation.join().unwrap();
    assert_eq!(
        acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .get(),
        initial.get() + 1
    );
    waiter.join().unwrap();
}

#[test]
fn bulk_write_snapshot_waits_for_active_native_execution_without_advancing_mapping_epoch() {
    let memory = shared_memory();
    let initial = memory.mapping_epoch();
    let active = memory.acquire_execution_lease();
    let (notified_tx, notified_rx) = mpsc::channel();
    memory.set_transition_notifier(Some(Arc::new(move || {
        let _ = notified_tx.send(());
    })));
    let (finished_tx, finished_rx) = mpsc::channel();
    let writer_memory = Arc::clone(&memory);
    let writer = thread::spawn(move || {
        let result = writer_memory.write_bytes(SPACE, PRIMARY, &[0x5a]);
        finished_tx.send(result).unwrap();
    });

    notified_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
    drop(active);
    finished_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    writer.join().unwrap();

    let mut observed = [0];
    memory.read_bytes(SPACE, PRIMARY, &mut observed).unwrap();
    assert_eq!(observed, [0x5a]);
    assert_eq!(memory.mapping_epoch(), initial);
}

#[test]
fn concurrent_alias_access_remains_coherent_without_torn_values() {
    let memory = shared_memory();
    let start = Arc::new(Barrier::new(3));
    let access = MemoryAccess::normal(MemoryAccessSize::Doubleword);
    let mut workers = Vec::new();
    for (address, value) in [
        (PRIMARY, MemoryValue::U64(0xaaaa_aaaa_aaaa_aaaa)),
        (ALIAS, MemoryValue::U64(0x5555_5555_5555_5555)),
    ] {
        let memory = Arc::clone(&memory);
        let start = Arc::clone(&start);
        workers.push(thread::spawn(move || {
            start.wait();
            for _ in 0..2_000 {
                memory.write(SPACE, address, access, value).unwrap();
                let observed = memory.read(SPACE, address, access).unwrap().value;
                assert!(matches!(
                    observed,
                    MemoryValue::U64(0xaaaa_aaaa_aaaa_aaaa)
                        | MemoryValue::U64(0x5555_5555_5555_5555)
                ));
            }
        }));
    }
    start.wait();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn failed_mapping_update_does_not_advance_the_epoch() {
    let memory = shared_memory();
    let initial = memory.mapping_epoch();
    assert!(
        memory
            .resize_zeroed_mapping(
                SPACE,
                PRIMARY,
                0,
                SYNTHETIC_PAGE_SIZE as u64,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Heap,
            )
            .is_err()
    );
    assert_eq!(memory.mapping_epoch(), initial);
}

#[test]
fn contending_exclusive_stores_allow_only_one_generation_winner() {
    let memory = shared_memory();
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::AcquireRelease,
        MemoryAccessClass::Exclusive,
    );
    memory
        .write(SPACE, PRIMARY, access, MemoryValue::U32(0))
        .unwrap();
    let start = Arc::new(Barrier::new(3));
    let (result_tx, result_rx) = mpsc::channel();
    let mut workers = Vec::new();
    for value in [1_u32, 2] {
        let memory = Arc::clone(&memory);
        let start = Arc::clone(&start);
        let result_tx = result_tx.clone();
        workers.push(thread::spawn(move || {
            let (_, reservation) = memory.load_exclusive(SPACE, PRIMARY, access).unwrap();
            start.wait();
            let succeeded = memory
                .store_exclusive(SPACE, ALIAS, access, MemoryValue::U32(value), reservation)
                .unwrap()
                .1;
            result_tx.send(succeeded).unwrap();
        }));
    }
    start.wait();
    let winners = [result_rx.recv().unwrap(), result_rx.recv().unwrap()]
        .into_iter()
        .filter(|succeeded| *succeeded)
        .count();
    assert_eq!(winners, 1);
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn release_acquire_message_passing_never_observes_stale_data() {
    let memory = shared_memory();
    let data = PRIMARY;
    let flag = GuestVirtualAddress::new(PRIMARY.get() + 8);
    let relaxed = MemoryAccess::normal(MemoryAccessSize::Word);
    let release = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::Release,
        MemoryAccessClass::Atomic,
    );
    let acquire = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::Acquire,
        MemoryAccessClass::Atomic,
    );
    let start = Arc::new(Barrier::new(3));

    let producer_memory = Arc::clone(&memory);
    let producer_start = Arc::clone(&start);
    let producer = thread::spawn(move || {
        producer_start.wait();
        producer_memory
            .write(SPACE, data, relaxed, MemoryValue::U32(0xfeed_beef))
            .unwrap();
        producer_memory
            .write(SPACE, flag, release, MemoryValue::U32(1))
            .unwrap();
    });
    let consumer_memory = Arc::clone(&memory);
    let consumer_start = Arc::clone(&start);
    let consumer = thread::spawn(move || {
        consumer_start.wait();
        loop {
            if consumer_memory.read(SPACE, flag, acquire).unwrap().value == MemoryValue::U32(1) {
                break;
            }
            thread::yield_now();
        }
        assert_eq!(
            consumer_memory.read(SPACE, data, relaxed).unwrap().value,
            MemoryValue::U32(0xfeed_beef)
        );
    });
    start.wait();
    producer.join().unwrap();
    consumer.join().unwrap();
}

#[test]
fn atomic_rmw_contends_on_physical_identity_from_parallel_host_workers() {
    let memory = shared_memory();
    let access = MemoryAccess::new(
        MemoryAccessSize::Doubleword,
        MemoryAlignment::Natural,
        MemoryOrdering::AcquireRelease,
        MemoryAccessClass::Atomic,
    );
    memory
        .write(SPACE, PRIMARY, access, MemoryValue::U64(0))
        .unwrap();
    let start = Arc::new(Barrier::new(5));
    let mut workers = Vec::new();
    for worker_index in 0..4 {
        let memory = Arc::clone(&memory);
        let start = Arc::clone(&start);
        workers.push(thread::spawn(move || {
            let address = if worker_index & 1 == 0 {
                PRIMARY
            } else {
                ALIAS
            };
            start.wait();
            for _ in 0..2_000 {
                memory
                    .atomic_read_modify_write(
                        SPACE,
                        address,
                        access,
                        AtomicRmwKind::Add,
                        MemoryValue::U64(1),
                    )
                    .unwrap();
            }
        }));
    }
    start.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        memory.read(SPACE, PRIMARY, access).unwrap().value,
        MemoryValue::U64(8_000)
    );
}

#[test]
fn compare_exchange_and_exclusives_cover_every_supported_scalar_width() {
    let memory = shared_memory();
    for (index, (size, initial, replacement)) in [
        (
            MemoryAccessSize::Byte,
            MemoryValue::U8(1),
            MemoryValue::U8(2),
        ),
        (
            MemoryAccessSize::Halfword,
            MemoryValue::U16(3),
            MemoryValue::U16(4),
        ),
        (
            MemoryAccessSize::Word,
            MemoryValue::U32(5),
            MemoryValue::U32(6),
        ),
        (
            MemoryAccessSize::Doubleword,
            MemoryValue::U64(7),
            MemoryValue::U64(8),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let address = PRIMARY.wrapping_add((index * 16) as u64);
        let alias = ALIAS.wrapping_add((index * 16) as u64);
        let exclusive = MemoryAccess::new(
            size,
            MemoryAlignment::Natural,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Exclusive,
        );
        let atomic = MemoryAccess::new(
            size,
            MemoryAlignment::Natural,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Atomic,
        );
        memory.write(SPACE, address, exclusive, initial).unwrap();
        let (_, reservation) = memory.load_exclusive(SPACE, address, exclusive).unwrap();
        assert!(
            memory
                .store_exclusive(SPACE, alias, exclusive, replacement, reservation)
                .unwrap()
                .1
        );
        let compared = memory
            .atomic_compare_exchange(SPACE, address, atomic, replacement, initial)
            .unwrap();
        assert_eq!(compared.previous, replacement);
        assert!(compared.stored);
        let failed = memory
            .atomic_compare_exchange(SPACE, alias, atomic, replacement, initial)
            .unwrap();
        assert_eq!(failed.previous, initial);
        assert!(!failed.stored);
    }
}

#[test]
fn atomic_code_writes_wait_for_instruction_cache_maintenance() {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(77);
    assert!(memory.add_ram_page(page));
    assert!(memory.initialize_ram(page, 0, &1_u32.to_le_bytes()));
    assert!(memory.map_page(SPACE, PRIMARY, page, MemoryPermissions::READ_WRITE_EXECUTE,));
    let cursor = memory.invalidation_cursor();
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::AcquireRelease,
        MemoryAccessClass::Atomic,
    );
    let failed = memory
        .atomic_compare_exchange(
            SPACE,
            PRIMARY,
            access,
            MemoryValue::U32(2),
            MemoryValue::U32(3),
        )
        .unwrap();
    assert!(!failed.stored);
    assert_eq!(memory.invalidation_cursor(), cursor);

    memory
        .atomic_read_modify_write(
            SPACE,
            PRIMARY,
            access,
            AtomicRmwKind::Add,
            MemoryValue::U32(1),
        )
        .unwrap();
    let mut records = Vec::new();
    let after_write = memory
        .read_invalidations_since(cursor, &mut records)
        .unwrap();
    assert_eq!(after_write, cursor);
    assert!(records.is_empty());

    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(PRIMARY),
        )
        .unwrap();
    memory
        .read_invalidations_since(after_write, &mut records)
        .unwrap();
    assert!(records.iter().any(|record| {
        matches!(
            record.kind,
            MemoryInvalidationKind::ExecutableContent {
                first,
                second: None
            } if first == page
        )
    }));
}
