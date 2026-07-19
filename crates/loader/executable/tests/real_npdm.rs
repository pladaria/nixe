//! Opt-in NPDM integration test against a caller-owned NSP.

use std::env;
use std::sync::Arc;

use swiitx_loader_content::{
    ExeFsLoader, NcaContentType, NcaKeySet, NcaLoader, NcaSectionType, NspLoader,
};
use swiitx_loader_executable::NpdmLoader;
use swiitx_loader_storage::{FileStorage, FormatLoader, StorageRef};

#[test]
#[ignore = "requires SWIITX_REAL_NSP and caller-owned keys"]
fn loads_npdm_from_real_nsp() {
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

    let mut loaded = 0;
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
            let Some(entry) = exefs.main_npdm() else {
                continue;
            };
            let npdm = NpdmLoader::load(exefs.open_entry(entry).expect("open main.npdm"))
                .expect("parse real main.npdm");
            let (minimum, maximum) = npdm.program_id_range();
            assert!((minimum..=maximum).contains(&npdm.program_id()));
            assert!(npdm.main_thread_priority() <= 0x3f);
            eprintln!(
                "{}: program_id={:#018x}, name={:?}, services={}, capabilities={}",
                entry.name(),
                npdm.program_id(),
                npdm.name_str(),
                npdm.requested_services().entries().len(),
                npdm.requested_kernel_capabilities().entries().len(),
            );
            loaded += 1;
        }
    }
    assert!(
        loaded != 0,
        "no loadable Program ExeFS/main.npdm found; failures: {failures:#?}"
    );
}
