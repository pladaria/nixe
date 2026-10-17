use std::sync::{Arc, mpsc};
use std::time::Duration;

use nixe_memory::*;

struct Readback {
    gate: ExecutionGate,
    neighbor: CanonicalBackingPage,
}

impl VisibilityCoordinator for Readback {
    fn cache_cpu_page(
        &self,
        _: DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        // A real demand may change mappings and access another captured page.
        // Neither mapping admission nor its earlier CPU guard may survive here.
        let _exclusive = self.gate.acquire_exclusive();
        let mut byte = [0];
        self.neighbor.read(0, &mut byte).unwrap();
        assert_eq!(byte, [0x11]);
        Ok(vec![0x22; request.size].into_boxed_slice())
    }
}

#[test]
fn cpu_word_reads_release_mapping_and_neighbor_locks_before_gpu_demand() {
    let gate = ExecutionGate::new();
    let store = CanonicalBackingStore::allocate_with_execution_gate(gate.clone()).unwrap();
    let allocation =
        CanonicalAllocation::zeroed_in(store, GuestPhysicalPageId::new(1), 8192, 4096).unwrap();
    allocation.write(0, &[0x11; 8]).unwrap();
    let full = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let gpu = full.snapshot_subrange(4096, 4096).unwrap();
    let mixed = CanonicalBackingRange::concatenate(vec![
        full.snapshot_subrange(0, 3).unwrap(),
        full.snapshot_subrange(4096, 5).unwrap(),
        full.snapshot_subrange(4, 3).unwrap(),
    ])
    .unwrap();
    let arena = DirectArena::new(12288).unwrap();
    for (index, page) in allocation.pages().iter().enumerate() {
        let backing = page.direct_backing().unwrap();
        let address = (index as u64 + 1) * 4096;
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: address,
                backing: &backing,
                protection: DirectProtection::ReadWrite,
            }])
            .unwrap();
        page.register_direct_alias(&arena, address, DirectProtection::ReadWrite)
            .unwrap();
    }
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(Readback {
        gate: gate.clone(),
        neighbor: allocation.pages()[0].clone(),
    });
    let write = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    CanonicalBackingRange::prepare_resident_device_accesses([(&gpu, write)], coordinator.as_ref())
        .unwrap();
    CanonicalBackingRange::publish_device_writes([(&gpu, write)], coordinator).unwrap();
    let epoch = gate.epoch();
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut bytes = [0; 11];
        mixed.read_word_stream(0, &mut bytes).unwrap();
        tx.send(bytes).unwrap();
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        [
            0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22, 0x22, 0x11, 0x11, 0x11,
        ]
    );
    reader.join().unwrap();
    assert_eq!(gate.epoch(), epoch);
    assert_eq!(arena.protection_at(4096), Some(DirectProtection::ReadWrite));
    assert_eq!(arena.protection_at(8192), Some(DirectProtection::Read));
}
