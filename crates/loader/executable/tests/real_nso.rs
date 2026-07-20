//! Opt-in integration test against caller-owned title content.

use std::env;
use std::sync::Arc;

use swiitx_loader_content::{
    ExeFsLoader, NcaContentType, NcaKeySet, NcaLoader, NcaSectionType, NspLoader, XciLoader,
};
use swiitx_loader_executable::{
    ExecutableFormat, ExternalSymbol, NsoLoader, NsoSegmentCompression, PreparationConfig,
    SymbolResolution,
};
use swiitx_loader_storage::{FileStorage, FormatLoader, StorageRef};

fn deterministic_runtime_export(symbol: ExternalSymbol<'_>) -> SymbolResolution {
    let hash = symbol
        .name()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
    SymbolResolution::Address(0x6000_0000 + (hash & 0x0fff_ffff))
}

#[test]
#[ignore = "requires SWIITX_REAL_PACKAGE and caller-owned keys"]
fn loads_main_nso_from_real_package() {
    let package = env::var_os("SWIITX_REAL_PACKAGE")
        .or_else(|| env::var_os("SWIITX_REAL_NSP"))
        .expect("set SWIITX_REAL_PACKAGE to an NSP or XCI path");
    let keys_dir = env::var_os("SWIITX_KEYS_DIR").expect("set SWIITX_KEYS_DIR to a keys directory");
    let keys_dir = std::path::PathBuf::from(keys_dir);
    let title_keys = keys_dir.join("title.keys");
    let mut keys = NcaKeySet::from_files(
        keys_dir.join("prod.keys"),
        title_keys.is_file().then_some(title_keys.as_path()),
    )
    .expect("load caller-owned keys");
    let path = std::path::PathBuf::from(package);
    let source: StorageRef = Arc::new(FileStorage::open(&path).expect("open package"));
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let entries: Vec<(String, StorageRef)> = if extension.eq_ignore_ascii_case("xci") {
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
    };
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

    let require_zbic = env::var_os("SWIITX_REQUIRE_ZBIC").is_some();
    let mut failures = Vec::new();
    let mut loaded = 0;
    let mut found_zbic = false;
    for (entry_name, storage) in entries.iter().filter(|(name, _)| name.ends_with(".nca")) {
        let nca = match NcaLoader::load_with_key_provider(storage.clone(), &keys) {
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
            let image = NsoLoader::load(exefs.open_entry(main).expect("open main NSO"))
                .expect("load real main NSO");
            assert_eq!(image.executable().format(), ExecutableFormat::Nso);
            assert_eq!(image.executable().segments().len(), 3);
            assert!(!image.executable().module_id().iter().all(|byte| *byte == 0));
            let prepared = image
                .prepare(
                    PreparationConfig {
                        image_base: 0x7100_0000,
                        address_limit: 0x8000_0000,
                    },
                    &deterministic_runtime_export,
                )
                .expect("prepare and relocate real main NSO");
            assert!(
                prepared
                    .mapping_at(prepared.entry_address())
                    .expect("entry mapping")
                    .permissions()
                    .is_executable()
            );
            assert!(prepared.mappings().iter().all(|mapping| {
                !(mapping.permissions().is_writable() && mapping.permissions().is_executable())
            }));
            let repeated = image
                .prepare(
                    PreparationConfig {
                        image_base: 0x7100_0000,
                        address_limit: 0x8000_0000,
                    },
                    &deterministic_runtime_export,
                )
                .expect("repeat deterministic preparation");
            assert_eq!(prepared, repeated);

            let dependency = exefs
                .entries()
                .iter()
                .filter(|candidate| candidate.name() != "main" && candidate.name() != "main.npdm")
                .find_map(|candidate| {
                    let source = exefs.open_entry(candidate).ok()?;
                    NsoLoader::load(source)
                        .ok()
                        .map(|image| (candidate.name(), image))
                });
            if let Some((name, dependency)) = dependency {
                let dependency_base = prepared
                    .mappings()
                    .last()
                    .expect("prepared main has mappings")
                    .guest_end()
                    .checked_add(0x1000)
                    .expect("dependency placement overflows");
                let dependency = dependency
                    .prepare(
                        PreparationConfig {
                            image_base: dependency_base,
                            address_limit: 0x8000_0000,
                        },
                        &deterministic_runtime_export,
                    )
                    .expect("prepare real dependency NSO");
                assert!(prepared.mappings().iter().all(|left| {
                    dependency.mappings().iter().all(|right| {
                        left.guest_end() <= right.guest_address()
                            || right.guest_end() <= left.guest_address()
                    })
                }));
                eprintln!("prepared dependency {name}");
            }
            eprintln!(
                "{}: flags={:#x}, compression={:?}, text_permissions={:?}",
                entry_name,
                image.metadata().flags(),
                image.metadata().compression(),
                image.executable().segments()[0].permissions()
            );
            loaded += 1;
            found_zbic |= image
                .metadata()
                .compression()
                .contains(&NsoSegmentCompression::Zbic);
        }
    }
    assert!(
        loaded != 0,
        "no loadable Program ExeFS/main NSO found; failures: {failures:#?}"
    );
    if require_zbic {
        assert!(found_zbic, "no ZBIC-compressed main NSO found in package");
    }
}
