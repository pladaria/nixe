use std::sync::Arc;

use nixe_loader_executable::{
    ExternalSymbol, NroLoader, NsoLoader, PreparationConfig, PreparedModule, SymbolResolution,
};
use nixe_loader_storage::{FormatLoader, Storage, StorageError, StorageRef};
use nixe_memory::{
    Access, AccessResult, AddressWidth, GuestVirtualAddress, Machine, MemoryPermissions, PAGE_SIZE,
};
use nixe_runtime::{
    install_prepared_module,
    memory::{MemoryKind, ProcessMemory},
};

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

fn memory(machine: &Machine) -> ProcessMemory {
    ProcessMemory::new(
        machine.account(16 * PAGE_SIZE).unwrap(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits32,
        nixe_runtime::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap()
}
fn read(memory: &ProcessMemory, address: u64, bytes: usize) -> Vec<u8> {
    let mut result = vec![0; bytes];
    assert!(matches!(
        memory
            .space()
            .read_into(GuestVirtualAddress::new(address), &mut result)
            .unwrap(),
        Access::Ready(())
    ));
    result
}
fn fetch(memory: &ProcessMemory, address: u64) -> u32 {
    let ranges = memory
        .space()
        .capture(
            GuestVirtualAddress::new(address),
            4,
            MemoryPermissions::EXECUTE,
        )
        .unwrap();
    let (result, _) = memory
        .account()
        .machine()
        .access_code_captured(ranges)
        .unwrap();
    let Access::Ready(AccessResult::Read(bytes)) = result else {
        panic!("private code must be ready")
    };
    u32::from_le_bytes(bytes[..].try_into().unwrap())
}

#[test]
fn executable_segments_preserve_bytes_relocations_bss_permissions_and_holes() {
    let machine = Machine::new(32 * PAGE_SIZE).unwrap();
    let memory = memory(&machine);
    let first = prepare_nro(BASE);
    let second = prepare_nro(BASE + 0x10_0000);
    install_prepared_module(&memory, &first, true).unwrap();
    install_prepared_module(&memory, &second, true).unwrap();
    assert_eq!(fetch(&memory, first.entry_address()), 0xd503_201f);
    assert_eq!(
        read(&memory, BASE + 0x3000, 4),
        0x2222_2222_u32.to_le_bytes()
    );
    assert_eq!(
        read(&memory, BASE + 0x5000, 8),
        (BASE + 0x123).to_le_bytes()
    );
    assert!(matches!(
        memory
            .space()
            .write_value(GuestVirtualAddress::new(BASE + 0x5000), 4, 0xaabb_ccdd)
            .unwrap(),
        Access::Ready(())
    ));
    assert_eq!(
        read(&memory, BASE + 0x5000, 4),
        0xaabb_ccdd_u32.to_le_bytes()
    );
    assert_eq!(read(&memory, BASE + 0x6000, 0x1000), vec![0; 0x1000]);
    for address in [BASE, BASE + 0x3000] {
        assert!(
            memory
                .space()
                .write_value(GuestVirtualAddress::new(address), 1, 1)
                .is_err()
        );
    }
    assert!(
        memory
            .space()
            .capture(
                GuestVirtualAddress::new(BASE + 0x5000),
                4,
                MemoryPermissions::EXECUTE
            )
            .is_err()
    );
    assert!(
        memory
            .query(GuestVirtualAddress::new(BASE + 0x2000))
            .unwrap()
            .mapping
            .is_none()
    );
    let text = memory
        .query(GuestVirtualAddress::new(BASE))
        .unwrap()
        .mapping
        .unwrap();
    assert_eq!(text.kind, MemoryKind::ModuleCode);
    assert_eq!(text.permissions, MemoryPermissions::READ_EXECUTE);
    assert_eq!(
        memory
            .query(GuestVirtualAddress::new(BASE + 0x5000))
            .unwrap()
            .mapping
            .unwrap()
            .kind,
        MemoryKind::MutableModuleCode
    );
    let mut identities = std::collections::BTreeSet::new();
    for module in [&first, &second] {
        for segment in module.mappings() {
            for address in
                (segment.guest_address()..segment.guest_end()).step_by(PAGE_SIZE as usize)
            {
                let translation = memory
                    .space()
                    .translate(GuestVirtualAddress::new(address), MemoryPermissions::NONE)
                    .unwrap();
                assert!(identities.insert(translation.physical));
            }
        }
    }
    assert_eq!(
        memory.account().used_bytes(),
        identities.len() as u64 * PAGE_SIZE
    );
    memory.close().unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn execute_only_text_fetches_but_rejects_data_reads() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let memory = memory(&machine);
    let module = prepare_execute_only_nso(BASE);
    install_prepared_module(&memory, &module, false).unwrap();
    assert_eq!(fetch(&memory, module.entry_address()), 0xd503_201f);
    assert!(
        memory
            .space()
            .read_value(GuestVirtualAddress::new(module.entry_address()), 4)
            .is_err()
    );
    let text = memory
        .query(GuestVirtualAddress::new(BASE))
        .unwrap()
        .mapping
        .unwrap();
    assert_eq!(text.permissions, MemoryPermissions::EXECUTE);
    assert_eq!(text.kind, MemoryKind::Code);
}

#[test]
fn failed_construction_releases_earlier_segments_and_cannot_change_another_process() {
    let machine = Machine::new(32 * PAGE_SIZE).unwrap();
    let module = prepare_nro(BASE);
    let first = memory(&machine);
    install_prepared_module(&first, &module, false).unwrap();
    let used = machine.statistics().allocated_bytes;
    let original = first
        .space()
        .translate(GuestVirtualAddress::new(BASE), MemoryPermissions::EXECUTE)
        .unwrap();
    {
        let second = ProcessMemory::new(
            machine.account(3 * PAGE_SIZE).unwrap(),
            machine.account(machine.arena().capacity()).unwrap(),
            AddressWidth::Bits32,
            nixe_runtime::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
            0,
        )
        .unwrap();
        assert!(install_prepared_module(&second, &module, false).is_err());
        assert_eq!(fetch(&first, BASE), 0xd503_201f);
    }
    assert_eq!(machine.statistics().allocated_bytes, used);
    let collision = install_prepared_module(&first, &module, false).unwrap_err();
    assert_eq!(collision.module_id(), module.module_id());
    assert_eq!(machine.statistics().allocated_bytes, used);
    let after = first
        .space()
        .translate(GuestVirtualAddress::new(BASE), MemoryPermissions::EXECUTE)
        .unwrap();
    assert_eq!(after.physical, original.physical);
    assert_eq!(after.allocation, original.allocation);
    first.close().unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn a_loader_prepared_image_executes_on_the_canonical_native_process() {
    use nixe_cpu::{
        execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState},
        platform::TargetPlatform,
        profile::ProcessCpuContext,
        state::a64::{A64GeneralRegister, A64Register},
    };
    use nixe_cpu_jit::{JitProcess, JitRunRequest, ThreadState};
    struct Timer;
    impl ArchitecturalTimer for Timer {
        fn snapshot(&self) -> TimerSnapshot {
            TimerSnapshot {
                counter: 0,
                frequency: 19_200_000,
            }
        }
    }
    let mut bytes = synthetic_nro();
    put_u32(&mut bytes, 0, 0x14000020); // B to the program after the header.
    put_u32(&mut bytes, 0x80, 0xd28000e0); // MOV X0,#7.
    put_u32(&mut bytes, 0x84, 0xd40000e1); // SVC #7.
    let image = NroLoader::load(storage(bytes))
        .unwrap()
        .prepare(
            PreparationConfig {
                image_base: BASE,
                address_limit: BASE + 0x10_0000,
            },
            &unresolved,
        )
        .unwrap();
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let memory = memory(&machine);
    install_prepared_module(&memory, &image, true).unwrap();
    let space = memory.space();
    let process = JitProcess::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, space.identity()),
        Arc::clone(space),
        19_200_000,
    )
    .unwrap();
    let mut thread = process.create_thread().unwrap();
    let mut state = ThreadState::default();
    state.set_pc(image.entry_address());
    let report = thread
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut state,
                instruction_budget: 1000,
                timer: &Timer,
                events: &VcpuEventState::default(),
                capture_context: false,
            },
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::SupervisorCall { immediate: 7, .. }
    ));
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
        7
    );
    assert!(process.try_shutdown().unwrap());
    memory.close().unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0);
}
