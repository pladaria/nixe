//! Opt-in integration test against caller-owned title content.

use std::env;
use std::sync::Arc;

use swiitx_loader_content::{
    ExeFsLoader, NcaContentType, NcaKeySet, NcaLoader, NcaSectionType, NspLoader,
};
use swiitx_loader_executable::{ExecutableFormat, NsoLoader};
use swiitx_loader_storage::{FileStorage, FormatLoader, StorageRef};

#[test]
#[ignore = "requires SWIITX_REAL_NSP and caller-owned keys"]
fn loads_main_nso_from_real_nsp() {
    let package = env::var_os("SWIITX_REAL_NSP").expect("set SWIITX_REAL_NSP to an NSP path");
    let keys_dir = env::var_os("SWIITX_KEYS_DIR").expect("set SWIITX_KEYS_DIR to a keys directory");
    let keys_dir = std::path::PathBuf::from(keys_dir);
    let title_keys = keys_dir.join("title.keys");
    let mut keys = NcaKeySet::from_files(
        keys_dir.join("prod.keys"),
        title_keys.is_file().then_some(title_keys.as_path()),
    )
    .expect("load caller-owned keys");
    let source: StorageRef = Arc::new(FileStorage::open(package).expect("open NSP"));
    let nsp = NspLoader::load(source).expect("parse NSP");
    for ticket in nsp
        .entries()
        .iter()
        .filter(|entry| entry.name().ends_with(".tik") && entry.size() >= 0x2b0)
    {
        let ticket = nsp.open_entry(ticket).expect("open ticket");
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

    let mut failures = Vec::new();
    for entry in nsp
        .entries()
        .iter()
        .filter(|entry| entry.name().ends_with(".nca"))
    {
        let storage = nsp.open_entry(entry).expect("open NCA entry");
        let nca = match NcaLoader::load_with_key_provider(storage, &keys) {
            Ok(nca) if nca.header().content_type() == NcaContentType::Program => nca,
            Ok(_) => continue,
            Err(error) => {
                failures.push(format!("{}: {error}", entry.name()));
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
                    failures.push(format!("{} ExeFS: {error}", entry.name()));
                    continue;
                }
            };
            let Some(main) = exefs.main() else { continue };
            let image = NsoLoader::load(exefs.open_entry(main).expect("open main NSO"))
                .expect("load real main NSO");
            assert_eq!(image.executable().format(), ExecutableFormat::Nso);
            assert_eq!(image.executable().segments().len(), 3);
            assert!(!image.executable().module_id().iter().all(|byte| *byte == 0));
            return;
        }
    }
    panic!("no loadable Program ExeFS/main NSO found; failures: {failures:#?}");
}
