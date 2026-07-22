//! Opt-in runtime integration against caller-owned title content.

use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nixe_cpu::address::{AddressSpaceId, GuestVirtualAddress};
use nixe_cpu::memory::{
    CpuMemory, DataAccessFaultReason, InstructionMemory, MemoryAccess, MemoryAccessSize,
    MemoryPermissions as CpuPermissions, MemoryValue, SYNTHETIC_PAGE_SIZE, SyntheticMemory,
};
use nixe_loader_content::{
    ExeFsLoader, NcaContentType, NcaKeySet, NcaLoader, NcaSectionType, NspLoader, XciLoader,
};
use nixe_loader_executable::{
    ExternalSymbol, MemoryPermissions as LoaderPermissions, NsoLoader, PreparationConfig,
    PreparedModule, SymbolResolution,
};
use nixe_loader_storage::{FileStorage, FormatLoader, StorageRef};
use nixe_runtime::install_prepared_module;

const SPACE: AddressSpaceId = AddressSpaceId::new(0x100);

fn deterministic_runtime_export(symbol: ExternalSymbol<'_>) -> SymbolResolution {
    let hash = symbol
        .name()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
    SymbolResolution::Address(0x6000_0000 + (hash & 0x0fff_ffff))
}

fn package_entries(path: &Path) -> Vec<(String, StorageRef)> {
    let source: StorageRef = Arc::new(FileStorage::open(path).expect("open package"));
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if extension.eq_ignore_ascii_case("xci") {
        let xci = XciLoader::load(source).expect("parse XCI");
        let secure = xci.secure_partition().expect("open XCI secure partition");
        secure
            .archive()
            .entries()
            .iter()
            .map(|entry| {
                (
                    entry.name().to_owned(),
                    secure
                        .open(entry.name())
                        .expect("open XCI entry")
                        .expect("listed XCI entry"),
                )
            })
            .collect()
    } else {
        let nsp = NspLoader::load(source).expect("parse NSP");
        nsp.entries()
            .iter()
            .map(|entry| {
                (
                    entry.name().to_owned(),
                    nsp.open_entry(entry).expect("open NSP entry"),
                )
            })
            .collect()
    }
}

fn load_keys(keys_dir: &Path, entries: &[(String, StorageRef)]) -> NcaKeySet {
    let title_keys = keys_dir.join("title.keys");
    let mut keys = NcaKeySet::from_files(
        keys_dir.join("prod.keys"),
        title_keys.is_file().then_some(title_keys.as_path()),
    )
    .expect("load caller-owned keys");
    for (_, ticket) in entries.iter().filter(|(name, storage)| {
        name.ends_with(".tik") && storage.len().is_ok_and(|size| size >= 0x2b0)
    }) {
        let mut encrypted_title_key = [0; 16];
        let mut rights_id = [0; 16];
        ticket
            .read_at(0x180, &mut encrypted_title_key)
            .expect("read encrypted title key");
        ticket
            .read_at(0x2a0, &mut rights_id)
            .expect("read ticket rights ID");
        keys.insert_encrypted_title_key(rights_id, encrypted_title_key);
    }
    keys
}

fn prepare_modules(
    entries: &[(String, StorageRef)],
    keys: &NcaKeySet,
) -> (PreparedModule, PreparedModule) {
    let mut failures = Vec::new();
    for (entry_name, storage) in entries.iter().filter(|(name, _)| name.ends_with(".nca")) {
        let nca = match NcaLoader::load_with_key_provider(storage.clone(), keys) {
            Ok(nca) if nca.header().content_type() == NcaContentType::Program => nca,
            Ok(_) => continue,
            Err(error) => {
                failures.push(format!("{entry_name}: {error}"));
                continue;
            }
        };
        for section in nca
            .sections()
            .iter()
            .filter(|section| section.section_type() == NcaSectionType::Pfs0)
        {
            let exefs = match ExeFsLoader::load_nca_section(section) {
                Ok(exefs) => exefs,
                Err(error) => {
                    failures.push(format!("{entry_name} ExeFS: {error}"));
                    continue;
                }
            };
            let Some(main) = exefs.main() else { continue };
            let Some(dependency_entry) = exefs
                .entries()
                .iter()
                .filter(|entry| entry.name() != "main" && entry.name() != "main.npdm")
                .find(|entry| {
                    exefs
                        .open_entry(entry)
                        .ok()
                        .and_then(|source| NsoLoader::load(source).ok())
                        .is_some()
                })
            else {
                failures.push(format!("{entry_name}: no dependency NSO"));
                continue;
            };
            let main = NsoLoader::load(exefs.open_entry(main).expect("open main NSO"))
                .expect("load main NSO")
                .prepare(
                    PreparationConfig {
                        image_base: 0x7100_0000,
                        address_limit: 0x8000_0000,
                    },
                    &deterministic_runtime_export,
                )
                .expect("prepare main NSO");
            let dependency_base = main
                .mappings()
                .last()
                .expect("prepared main has mappings")
                .guest_end()
                .checked_add(SYNTHETIC_PAGE_SIZE as u64)
                .expect("dependency placement overflows");
            let dependency = NsoLoader::load(
                exefs
                    .open_entry(dependency_entry)
                    .expect("open dependency NSO"),
            )
            .expect("load dependency NSO")
            .prepare(
                PreparationConfig {
                    image_base: dependency_base,
                    address_limit: 0x8000_0000,
                },
                &deterministic_runtime_export,
            )
            .expect("prepare dependency NSO");
            return (main, dependency);
        }
    }
    panic!("no Program ExeFS with main and dependency NSOs; failures: {failures:#?}");
}

fn cpu_permissions(permissions: LoaderPermissions) -> CpuPermissions {
    match (
        permissions.is_readable(),
        permissions.is_writable(),
        permissions.is_executable(),
    ) {
        (true, false, false) => CpuPermissions::READ,
        (true, true, false) => CpuPermissions::READ_WRITE,
        (true, false, true) => CpuPermissions::READ_EXECUTE,
        (false, false, true) => CpuPermissions::EXECUTE,
        combination => panic!("unsupported prepared permissions: {combination:?}"),
    }
}

fn verify_module(memory: &SyntheticMemory, module: &PreparedModule, pages: &mut BTreeSet<u64>) {
    for mapping in module.mappings() {
        let expected_permissions = cpu_permissions(mapping.permissions());
        for (index, bytes) in mapping
            .bytes()
            .chunks_exact(SYNTHETIC_PAGE_SIZE)
            .enumerate()
        {
            let address = GuestVirtualAddress::new(
                mapping.guest_address() + (index * SYNTHETIC_PAGE_SIZE) as u64,
            );
            let info = memory.mapping_info(SPACE, address).expect("installed page");
            assert_eq!(info.permissions, expected_permissions);
            assert!(pages.insert(info.physical_page.get()));
            assert_ne!(info.permissions, CpuPermissions::READ_WRITE_EXECUTE);
            if mapping.permissions().is_readable() {
                assert_eq!(
                    memory
                        .read(SPACE, address, MemoryAccess::normal(MemoryAccessSize::Byte),)
                        .expect("read installed byte")
                        .value,
                    MemoryValue::U8(bytes[0])
                );
            }
            if mapping.permissions().is_executable() {
                assert_eq!(
                    memory
                        .fetch32(SPACE, address)
                        .expect("fetch installed code")
                        .bits,
                    u32::from_le_bytes(bytes[..4].try_into().expect("instruction bytes"))
                );
                if !mapping.permissions().is_readable() {
                    assert_eq!(
                        memory
                            .read(SPACE, address, MemoryAccess::normal(MemoryAccessSize::Byte),)
                            .expect_err("execute-only page must reject data reads")
                            .reason,
                        DataAccessFaultReason::ReadPermissionDenied
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NIXE_REAL_PACKAGE and caller-owned keys"]
fn installs_real_main_and_dependency_into_runtime_memory() {
    let package = env::var_os("NIXE_REAL_PACKAGE")
        .or_else(|| env::var_os("NIXE_REAL_NSP"))
        .expect("set NIXE_REAL_PACKAGE to an NSP or XCI path");
    let keys_dir =
        PathBuf::from(env::var_os("NIXE_KEYS_DIR").expect("set NIXE_KEYS_DIR to a keys directory"));
    let entries = package_entries(Path::new(&package));
    let keys = load_keys(&keys_dir, &entries);
    let (main, dependency) = prepare_modules(&entries, &keys);
    assert!(main.mappings().iter().all(|left| {
        dependency.mappings().iter().all(|right| {
            left.guest_end() <= right.guest_address() || right.guest_end() <= left.guest_address()
        })
    }));

    let mut memory = SyntheticMemory::new();
    install_prepared_module(&mut memory, SPACE, &dependency).expect("install dependency");
    install_prepared_module(&mut memory, SPACE, &main).expect("install main");
    let fetched = memory
        .fetch32(SPACE, GuestVirtualAddress::new(main.entry_address()))
        .expect("fetch installed main entry");
    assert_eq!(
        fetched.bits,
        u32::from_le_bytes(
            main.read_guest(main.entry_address(), 4)
                .expect("prepared entry bytes")
                .try_into()
                .expect("instruction bytes")
        )
    );

    let mut physical_pages = BTreeSet::new();
    verify_module(&memory, &dependency, &mut physical_pages);
    verify_module(&memory, &main, &mut physical_pages);
}
