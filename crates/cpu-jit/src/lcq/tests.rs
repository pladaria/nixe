use super::*;
use crate::abi::FpSpecialization;
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryPermissions, ProcessMemory, SyntheticMemory,
};
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId};

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const NOP: u32 = 0xd503_201f;
fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}
fn memory(pages: u64) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    for index in 0..pages {
        let page = GuestPhysicalPageId::new(index + 1);
        assert!(memory.add_ram_page(page));
        assert!(memory.initialize_ram(page, 0, &NOP.to_le_bytes().repeat(1024)));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(index * 4096),
            page,
            MemoryPermissions::READ_EXECUTE
        ));
    }
    memory
}

#[test]
fn every_control_boundary_stops_without_fetching_its_successor() {
    for (bits, end) in [
        (0x1400_0001u32, End::Control),
        (0x9400_0001, End::Control),
        (0x5400_0020, End::Control),
        (0xb400_0020, End::Control),
        (0x3600_0020, End::Control),
        (0xd61f_0000, End::Control),
        (0xd63f_0000, End::Control),
        (0xd65f_03c0, End::Control),
        (0xd400_0021, End::Architectural),
        (0xd420_0020, End::Architectural),
        (0xd51b_4400, End::FpMode),
        (0xd51b_4420, End::Architectural),
        (0xd53b_4420, End::Architectural),
        (0xd503_203f, End::Architectural),
        (0xd503_205f, End::Architectural),
        (0xd503_207f, End::Architectural),
        (0xd503_209f, End::Architectural),
        (0xd503_20bf, End::Architectural), // SEVL
        (0xd53b_e000, End::Architectural), // timer frequency
        (0xd53b_e020, End::Architectural), // timer counter
        (0xd503_3bbf, End::Architectural), // DMB
        (0xd503_3f5f, End::Architectural), // CLREX
        (0xd508_751f, End::Architectural), // IC IALLU
        (0xd50b_7520, End::Architectural), // IC IVAU
        (0x1e62_0420, End::Architectural), // FCCMP
        (0x1e67_4020, End::Architectural), // FRINTX
        (0x1e67_c020, End::Architectural), // FRINTI
        (0x1e66_4020, End::Architectural), // FRINTA
        (0x1e20_0020, End::Architectural), // FCVTNS
        (0x9e59_c020, End::Architectural), // FCVTZU X0,D1,#16
        (0xd53b_0000, End::Unsupported),
    ] {
        let mut memory = memory(1);
        assert!(memory.initialize_ram(GuestPhysicalPageId::new(1), 4092, &bits.to_le_bytes()));
        let fragment = Fragment::capture(&memory, key(4092)).unwrap();
        assert_eq!(fragment.end, end, "{bits:08x}");
        assert_eq!(fragment.instructions.len(), 1);
        assert!(fragment.image.fault().is_none()); // Next page does not exist.
        assert_eq!(fragment.image.words()[0].bits, bits);
    }
}

#[test]
fn emergency_cut_crosses_pages_and_overlap_retains_both_dependencies() {
    let memory = memory(2);
    for pc in [2048, 2052, 2064] {
        let fragment = Fragment::capture(&memory, key(pc)).unwrap();
        assert_eq!(fragment.instructions.len(), 512);
        assert_eq!(
            fragment.end,
            End::Limit {
                continuation: GuestVirtualAddress::new(pc + 2048)
            }
        );
        assert_eq!(
            fragment.image.dependencies().count(),
            if pc == 2048 { 1 } else { 2 }
        );
    }
}

#[test]
fn demanded_fetch_failure_keeps_valid_prefix_and_exact_address() {
    let memory = memory(1);
    let fragment = Fragment::capture(&memory, key(4092)).unwrap();
    assert_eq!(fragment.instructions.len(), 1);
    assert_eq!(fragment.end, End::FetchFault);
    assert_eq!(
        fragment.image.fault().unwrap().address,
        GuestVirtualAddress::new(4096)
    );
    let empty = Fragment::capture(&memory, key(4096)).unwrap();
    assert!(empty.instructions.is_empty());
    assert_eq!(
        empty.image.fault().unwrap().address,
        GuestVirtualAddress::new(4096)
    );
}

#[test]
fn executable_alias_write_stales_owned_image_without_changing_its_bytes() {
    let mut memory = memory(1);
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x4000),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE
    ));
    let fragment = Fragment::capture(&memory, key(0)).unwrap();
    assert!(memory.image_is_current(&fragment.image));
    memory
        .write_bytes(
            SPACE,
            GuestVirtualAddress::new(0x4000),
            &0x1400_0000u32.to_le_bytes(),
        )
        .unwrap();
    assert!(!memory.image_is_current(&fragment.image));
    assert_eq!(fragment.image.words()[0].bits, NOP);
    assert_eq!(
        Fragment::capture(&memory, key(0))
            .unwrap()
            .instructions
            .len(),
        1
    );
}

#[test]
fn real_memory_capture_observes_host_alias_writes_and_owner_identity() {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    memory
        .initialize_ram(page, 0, &NOP.to_le_bytes().repeat(1024))
        .unwrap();
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x4000),
        page,
        MemoryPermissions::READ_WRITE
    ));
    let fragment = Fragment::capture(&memory, key(0)).unwrap();
    assert!(memory.image_is_current(&fragment.image));
    assert!(!ExecutionMemory::new().image_is_current(&fragment.image));
    memory
        .overwrite_mapped_ram(
            SPACE,
            GuestVirtualAddress::new(0x4000),
            &0x1400_0000u32.to_le_bytes(),
        )
        .unwrap();
    assert!(!memory.image_is_current(&fragment.image));
    assert_eq!(fragment.image.words()[0].bits, NOP);
    assert_eq!(
        Fragment::capture(&memory, key(0))
            .unwrap()
            .instructions
            .len(),
        1
    );
}

#[test]
fn captured_compilation_reserves_identity_and_releases_abandoned_claim() {
    let process = std::sync::Arc::new(
        crate::lifetime::Lifetime::new(crate::executable::Cache::new().unwrap()).unwrap(),
    );
    let mut reader = process.register().unwrap();
    let crate::lifetime::compile::Request::Owner(claim) = reader.claim(key(0)).unwrap() else {
        panic!()
    };
    let captured = Compilation::capture(claim, &memory(1)).unwrap();
    assert_eq!(captured.fragment.instructions.len(), 512);
    assert!(captured.identity.version().get() > 0);
    captured.claim.validate().unwrap();
    drop(captured);
    assert!(matches!(
        reader.claim(key(0)).unwrap(),
        crate::lifetime::compile::Request::Owner(_)
    ));
}

#[test]
fn raw_direct_alias_store_invalidates_capture_without_per_store_generations() {
    use nixe_cpu::memory::{DataAccessKind, DirectFaultResolution, MemoryAccessSize};
    use nixe_memory::{DirectBackendPolicy, DirectProtection, MemoryInvalidationSource};
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    memory
        .initialize_ram(page, 0, &0x1400_0000u32.to_le_bytes())
        .unwrap();
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x4000),
        page,
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, DirectBackendPolicy::Required)
        .unwrap();
    let fragment = Fragment::capture(&memory, key(0)).unwrap();
    let cursor = memory.invalidation_cursor();
    assert!(memory.image_is_current(&fragment.image));
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x4000)),
        Some(DirectProtection::Read)
    );
    {
        let _lease = memory.acquire_execution_lease();
        assert_eq!(
            memory.resolve_direct_fault(
                SPACE,
                GuestVirtualAddress::new(0x4000),
                MemoryAccessSize::Word,
                DataAccessKind::Write
            ),
            DirectFaultResolution::Retry
        );
        let view = memory.direct_address_space_view(SPACE).unwrap();
        // Same shared alias and execution lease as generated native stores.
        unsafe {
            (view.host_address(0x4000).unwrap() as *mut u32).write_volatile(NOP.to_le());
        }
    }
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert!(!memory.image_is_current(&fragment.image));
    let replacement = Fragment::capture(&memory, key(0)).unwrap();
    assert_eq!(replacement.image.words()[0].bits, NOP);
    assert!(memory.image_is_current(&replacement.image));
    assert_eq!(fragment.image.words()[0].bits, 0x1400_0000);
}

#[test]
fn two_page_capture_cannot_mix_a_concurrent_host_write_through_aliases() {
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;
    let mut memory = ExecutionMemory::new();
    for index in 0..2 {
        let page = GuestPhysicalPageId::new(index + 1);
        assert!(memory.add_ram_page(page));
        memory
            .initialize_ram(page, 0, &NOP.to_le_bytes().repeat(1024))
            .unwrap();
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(index * 4096),
            page,
            MemoryPermissions::READ_EXECUTE
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x4000 + index * 4096),
            page,
            MemoryPermissions::READ_WRITE
        ));
    }
    let memory = Arc::new(memory);
    let entered = Barrier::new(2);
    let release = Barrier::new(2);
    let (done, received) = mpsc::channel();
    std::thread::scope(|scope| {
        let memory = &memory;
        let entered = &entered;
        let release = &release;
        let capture = scope.spawn(move || {
            memory.capture_instructions(
                SPACE,
                GuestVirtualAddress::new(4092),
                NonZeroU16::new(2).unwrap(),
                &|pc, _| {
                    if pc.get() == 4092 {
                        entered.wait();
                        release.wait();
                    }
                    false
                },
            )
        });
        entered.wait();
        let writer = scope.spawn(move || {
            memory
                .overwrite_mapped_ram(
                    SPACE,
                    GuestVirtualAddress::new(0x4ffc),
                    &0x1400_0000u32.to_le_bytes().repeat(2),
                )
                .unwrap();
            done.send(()).unwrap();
        });
        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        release.wait();
        let image = capture.join().unwrap();
        writer.join().unwrap();
        assert_eq!(
            image
                .words()
                .iter()
                .map(|word| word.bits)
                .collect::<Vec<_>>(),
            [NOP, NOP]
        );
        assert!(!memory.image_is_current(&image));
        let next = memory.capture_instructions(
            SPACE,
            GuestVirtualAddress::new(4092),
            NonZeroU16::new(2).unwrap(),
            &|_, _| false,
        );
        assert!(next.words().iter().all(|word| word.bits == 0x1400_0000));
    });
}

#[test]
fn device_owned_code_reconciles_outside_capture_locks_then_restarts() {
    use nixe_memory::{
        CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
        DeviceVisibilityPoint, DeviceVisibilityRequest, NonCpuDeviceId, VisibilityCoordinator,
        VisibilityCoordinatorError,
    };
    use std::sync::{Arc, Weak};
    struct Device {
        memory: Weak<ExecutionMemory>,
    }
    impl VisibilityCoordinator for Device {
        fn make_device_visible(
            &self,
            _: DeviceVisibilityRequest,
            _: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }
        fn make_cpu_visible(
            &self,
            _: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            let memory = self.memory.upgrade().unwrap();
            let _lease = memory.acquire_execution_lease();
            // Both operations would deadlock if capture retained the gate or
            // mapping lock while calling the device coordinator.
            assert!(
                memory
                    .code_page_span(SPACE, GuestVirtualAddress::new(0))
                    .is_ok()
            );
            Ok(0x1400_0000u32.to_le_bytes().repeat(1024).into_boxed_slice())
        }
    }
    use nixe_cpu::memory::InstructionMemory;
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    memory
        .initialize_ram(page, 0, &NOP.to_le_bytes().repeat(1024))
        .unwrap();
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x4000),
        page,
        MemoryPermissions::READ_WRITE
    ));
    let memory = Arc::new(memory);
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x4000),
            4096,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let device: Arc<dyn VisibilityCoordinator> = Arc::new(Device {
        memory: Arc::downgrade(&memory),
    });
    let access = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    range.prepare_device_access(access, device.clone()).unwrap();
    range.publish_device_write(access, device).unwrap();
    let fragment = Fragment::capture(memory.as_ref(), key(0)).unwrap();
    assert_eq!(fragment.end, End::Control);
    assert_eq!(fragment.image.words()[0].bits, 0x1400_0000);
    assert!(memory.image_is_current(&fragment.image));
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0),
            4096,
            MemoryPermissions::READ,
        )
        .unwrap();
    assert!(!memory.image_is_current(&fragment.image));
    assert_eq!(
        Fragment::capture(memory.as_ref(), key(0)).unwrap().end,
        End::FetchFault
    );
}
