use std::sync::Arc;

use nixe_cpu::address::{AddressSpaceId, GuestVirtualAddress};
use nixe_cpu::error::InstructionFetchFaultReason;
use nixe_cpu::memory::{
    CpuMemory, DataAccessFaultReason, InstructionMemory, MemoryAccess, MemoryAccessSize,
    MemoryPermissions, MemoryValue, SyntheticInstallStage, SyntheticMemory,
};
use nixe_loader_executable::{
    ExternalSymbol, NroLoader, NsoLoader, PreparationConfig, PreparedModule, SymbolResolution,
};
use nixe_loader_storage::{FormatLoader, Storage, StorageError, StorageRef};
use nixe_runtime::{
    BackendInstallError, InstallStage, ModuleMemoryBackend, PageRequest, install_prepared_module,
};

const SPACE: AddressSpaceId = AddressSpaceId::new(9);
const BASE: u64 = 0x7100_0000;

#[derive(Debug)]
struct Bytes(Vec<u8>);

impl Storage for Bytes {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.0.len() as u64)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
        let end = start
            .checked_add(buffer.len())
            .ok_or(StorageError::OutOfBounds)?;
        buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
        Ok(())
    }
}

fn storage(bytes: Vec<u8>) -> StorageRef {
    Arc::new(Bytes(bytes))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_i64(bytes: &mut [u8], offset: usize, value: i64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn dynamic(bytes: &mut [u8], index: usize, tag: i64, value: u64) {
    let offset = 0x3100 + index * 16;
    put_i64(bytes, offset, tag);
    put_u64(bytes, offset + 8, value);
}

fn unresolved(_: ExternalSymbol<'_>) -> SymbolResolution {
    SymbolResolution::Unresolved
}

fn synthetic_nro() -> Vec<u8> {
    let mut bytes = vec![0; 0x6000];
    bytes[..0x2000].fill(0x11);
    bytes[0x3000..0x4000].fill(0x22);
    bytes[0x5000..0x6000].fill(0x33);
    bytes[..0x80].fill(0);
    bytes[..4].copy_from_slice(&0xd503_201f_u32.to_le_bytes());
    bytes[0x100..0x104].copy_from_slice(b"MOD0");
    put_i32(&mut bytes, 0x104, 0x3000);
    put_i32(&mut bytes, 0x108, 0x5f00);
    put_i32(&mut bytes, 0x10c, 0x6221);
    put_i32(&mut bytes, 0x110, 0);
    put_i32(&mut bytes, 0x114, 0);
    put_i32(&mut bytes, 0x118, 0x4f00);
    dynamic(&mut bytes, 0, 7, 0x3200);
    dynamic(&mut bytes, 1, 8, 24);
    dynamic(&mut bytes, 2, 9, 24);
    dynamic(&mut bytes, 3, 0x6fff_fff9, 1);
    dynamic(&mut bytes, 4, 5, 0x3300);
    dynamic(&mut bytes, 5, 10, 1);
    dynamic(&mut bytes, 6, 6, 0x3310);
    dynamic(&mut bytes, 7, 11, 24);
    dynamic(&mut bytes, 8, 0, 0);
    put_u64(&mut bytes, 0x3200, 0x5000);
    put_u64(&mut bytes, 0x3208, 1027);
    put_i64(&mut bytes, 0x3210, 0x123);
    bytes[0x10..0x14].copy_from_slice(b"NRO0");
    put_u32(&mut bytes, 0x04, 0x100);
    put_u32(&mut bytes, 0x18, 0x6000);
    put_u32(&mut bytes, 0x20, 0);
    put_u32(&mut bytes, 0x24, 0x2000);
    put_u32(&mut bytes, 0x28, 0x3000);
    put_u32(&mut bytes, 0x2c, 0x1000);
    put_u32(&mut bytes, 0x30, 0x5000);
    put_u32(&mut bytes, 0x34, 0x1000);
    put_u32(&mut bytes, 0x38, 0x321);
    bytes[0x40..0x60].fill(0x44);
    put_u32(&mut bytes, 0x70, 0x3300);
    put_u32(&mut bytes, 0x74, 1);
    put_u32(&mut bytes, 0x78, 0x3310);
    put_u32(&mut bytes, 0x7c, 24);
    bytes
}

fn synthetic_execute_only_nso() -> Vec<u8> {
    let mut bytes = vec![0; 0x2900];
    bytes[..4].copy_from_slice(b"NSO0");
    put_u32(&mut bytes, 0x0c, 1 << 6);
    for (descriptor, file, memory, size) in [
        (0x10, 0x100, 0, 0x1000),
        (0x20, 0x1100, 0x1000, 0x1000),
        (0x30, 0x2100, 0x2000, 0x800),
    ] {
        put_u32(&mut bytes, descriptor, file);
        put_u32(&mut bytes, descriptor + 4, memory);
        put_u32(&mut bytes, descriptor + 8, size);
    }
    put_u32(&mut bytes, 0x1c, 0x100);
    put_u32(&mut bytes, 0x3c, 0x321);
    bytes[0x40..0x60].fill(0x55);
    put_u32(&mut bytes, 0x60, 0x1000);
    put_u32(&mut bytes, 0x64, 0x1000);
    put_u32(&mut bytes, 0x68, 0x800);
    bytes[0x100..0x1100].fill(0x66);
    bytes[0x100..0x104].copy_from_slice(&0xd503_201f_u32.to_le_bytes());
    put_u32(&mut bytes, 0x104, 0);
    bytes[0x1100..0x2100].fill(0x77);
    bytes[0x2100..0x2900].fill(0x88);
    bytes
}

fn prepare_nro(base: u64) -> PreparedModule {
    NroLoader::load(storage(synthetic_nro()))
        .unwrap()
        .prepare(
            PreparationConfig {
                image_base: base,
                address_limit: base + 0x10_0000,
            },
            &unresolved,
        )
        .unwrap()
}

fn prepare_execute_only_nso(base: u64) -> PreparedModule {
    NsoLoader::load(storage(synthetic_execute_only_nso()))
        .unwrap()
        .prepare(
            PreparationConfig {
                image_base: base,
                address_limit: base + 0x10_0000,
            },
            &unresolved,
        )
        .unwrap()
}

#[test]
fn prepared_mappings_are_observable_only_through_cpu_memory_contracts() {
    let first = prepare_nro(BASE);
    let second = prepare_nro(BASE + 0x10_0000);
    let mut memory = SyntheticMemory::new();

    install_prepared_module(&mut memory, SPACE, &first).unwrap();
    install_prepared_module(&mut memory, SPACE, &second).unwrap();

    let fetched = memory
        .fetch32(SPACE, GuestVirtualAddress::new(first.entry_address()))
        .unwrap();
    assert_eq!(fetched.bits, 0xd503_201f);
    let read_only = memory
        .read(
            SPACE,
            GuestVirtualAddress::new(BASE + 0x3000),
            MemoryAccess::normal(MemoryAccessSize::Word),
        )
        .unwrap();
    assert_eq!(read_only.value, MemoryValue::U32(0x2222_2222));
    let writable = GuestVirtualAddress::new(BASE + 0x5000);
    assert_eq!(
        memory
            .read(
                SPACE,
                writable,
                MemoryAccess::normal(MemoryAccessSize::Doubleword)
            )
            .unwrap()
            .value,
        MemoryValue::U64(BASE + 0x123)
    );
    memory
        .write(
            SPACE,
            writable,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0xaabb_ccdd),
        )
        .unwrap();
    assert_eq!(
        memory
            .read(
                SPACE,
                writable,
                MemoryAccess::normal(MemoryAccessSize::Word)
            )
            .unwrap()
            .value,
        MemoryValue::U32(0xaabb_ccdd)
    );
    for address in [BASE + 0x6000, BASE + 0x6320, BASE + 0x6fff] {
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(address),
                    MemoryAccess::normal(MemoryAccessSize::Byte)
                )
                .unwrap()
                .value,
            MemoryValue::U8(0)
        );
    }
    assert_eq!(
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(BASE + 0x3000),
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(1)
            )
            .unwrap_err()
            .reason,
        DataAccessFaultReason::WritePermissionDenied
    );
    assert_eq!(
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(BASE),
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(1)
            )
            .unwrap_err()
            .reason,
        DataAccessFaultReason::WritePermissionDenied
    );
    assert_eq!(
        memory.fetch32(SPACE, writable).unwrap_err().reason,
        InstructionFetchFaultReason::ExecutePermissionDenied
    );

    let page_addresses = [
        BASE,
        BASE + 0x1000,
        BASE + 0x3000,
        BASE + 0x5000,
        BASE + 0x6000,
    ];
    let identities = page_addresses.map(|address| {
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(address))
            .unwrap()
            .physical_page
    });
    for (index, identity) in identities.iter().enumerate() {
        assert!(!identities[..index].contains(identity));
    }
    for address in page_addresses {
        assert_ne!(
            memory
                .mapping_info(SPACE, GuestVirtualAddress::new(address))
                .unwrap()
                .permissions,
            MemoryPermissions::READ_WRITE_EXECUTE
        );
    }
    assert_eq!(
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(BASE))
            .unwrap()
            .permissions,
        MemoryPermissions::READ_EXECUTE
    );
    assert!(
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(BASE + 0x2000))
            .is_none()
    );
    assert!(
        memory
            .mapping_info(SPACE, GuestVirtualAddress::new(second.entry_address()))
            .is_some()
    );
}

#[test]
fn execute_only_text_fetches_but_rejects_data_reads() {
    let module = prepare_execute_only_nso(BASE);
    let mut memory = SyntheticMemory::new();
    install_prepared_module(&mut memory, SPACE, &module).unwrap();
    let entry = GuestVirtualAddress::new(module.entry_address());

    assert_eq!(memory.fetch32(SPACE, entry).unwrap().bits, 0xd503_201f);
    assert_eq!(
        memory
            .read(SPACE, entry, MemoryAccess::normal(MemoryAccessSize::Word))
            .unwrap_err()
            .reason,
        DataAccessFaultReason::ReadPermissionDenied
    );
    assert_eq!(
        memory.mapping_info(SPACE, entry).unwrap().permissions,
        MemoryPermissions::EXECUTE
    );
}

#[test]
fn collisions_and_every_backend_stage_roll_back_the_complete_module() {
    let module = prepare_nro(BASE);
    let mut occupied = SyntheticMemory::new();
    install_prepared_module(&mut occupied, SPACE, &module).unwrap();
    let pages_before = occupied.physical_page_count();
    let dependency_before = occupied
        .fetch32(SPACE, GuestVirtualAddress::new(module.entry_address()))
        .unwrap()
        .dependencies;
    let collision = install_prepared_module(&mut occupied, SPACE, &module).unwrap_err();
    assert_eq!(collision.stage(), InstallStage::Preflight);
    assert_eq!(occupied.physical_page_count(), pages_before);
    assert_eq!(
        occupied
            .fetch32(SPACE, GuestVirtualAddress::new(module.entry_address()))
            .unwrap()
            .dependencies,
        dependency_before
    );

    for (synthetic_stage, expected_stage) in [
        (SyntheticInstallStage::Preflight, InstallStage::Preflight),
        (SyntheticInstallStage::Allocation, InstallStage::Allocation),
        (
            SyntheticInstallStage::Initialization,
            InstallStage::Initialization,
        ),
        (
            SyntheticInstallStage::Publication,
            InstallStage::Publication,
        ),
    ] {
        let mut memory = SyntheticMemory::new();
        memory.inject_install_failure(synthetic_stage, 1, "requested failure");
        let error = install_prepared_module(&mut memory, SPACE, &module).unwrap_err();
        assert_eq!(error.module_id(), module.module_id());
        assert_eq!(error.stage(), expected_stage);
        assert_eq!(error.cause(), "requested failure");
        assert_eq!(memory.physical_page_count(), 0);
        assert!(
            memory
                .mapping_info(SPACE, GuestVirtualAddress::new(BASE))
                .is_none()
        );
    }
}

struct GeometryBackend(usize);

impl ModuleMemoryBackend for GeometryBackend {
    fn page_size(&self) -> usize {
        self.0
    }

    fn install_pages_atomic(
        &mut self,
        _: AddressSpaceId,
        _: &[PageRequest<'_>],
    ) -> Result<(), BackendInstallError> {
        panic!("invalid geometry must be rejected before calling the backend")
    }
}

#[test]
fn incompatible_backend_page_geometry_fails_during_runtime_preflight() {
    let module = prepare_nro(BASE);
    for page_size in [0, 8192] {
        let error =
            install_prepared_module(&mut GeometryBackend(page_size), SPACE, &module).unwrap_err();
        assert_eq!(error.module_id(), module.module_id());
        assert_eq!(error.stage(), InstallStage::Preflight);
        assert!(error.cause().contains("page"));
    }
}
